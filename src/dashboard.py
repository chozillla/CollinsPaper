"""
Interactive Dashboard: HDM Detection Trial Results.

Generates a self-contained HTML file with full Plotly interactivity
(hover, zoom, pan, toggle traces). No server required — open directly
in a browser or host via GitHub Pages.

Usage:  uv run python src/dashboard.py
Output: results/dashboard.html
"""

import json
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
    return gemini, hotword, random_bl, annotations


def build_dashboard():
    gemini, hotword, random_bl, annotations = load_data()

    # ── Aggregate data ──────────────────────────────────────────────────────
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

    # ── Build figure ────────────────────────────────────────────────────────
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
        vertical_spacing=0.14,
        horizontal_spacing=0.14,
    )

    # Style subplot titles — nudge up so they sit above the chart area
    for ann in fig["layout"]["annotations"]:
        ann["font"] = dict(size=13, color="#333")
        ann["y"] = ann["y"] + 0.015

    # ────────────────────────────────────────────────────────────────────────
    # (1,1) F1 Comparison Bar
    # ────────────────────────────────────────────────────────────────────────
    for name, avg, std, _, color in methods:
        fig.add_trace(go.Bar(
            x=[name], y=[avg],
            error_y=dict(type="data", array=[std], visible=True, thickness=1.5),
            marker_color=color, name=name,
            hovertemplate=f"<b>{name}</b><br>F1: {avg:.3f} ± {std:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=1)
        # Place value label well above the error bar
        fig.add_annotation(
            x=name, y=avg + std + 0.04,
            text=f"<b>{avg:.3f}</b>", showarrow=False,
            font=dict(size=12, color=color),
            row=1, col=1,
        )
    # Paper reference
    fig.add_hline(y=0.87, line_dash="dash", line_color=C["paper"], line_width=2,
                  annotation_text="Collins et al. F1=0.87",
                  annotation_font_color=C["paper"], annotation_font_size=10,
                  row=1, col=1)

    # ────────────────────────────────────────────────────────────────────────
    # (1,2) Per-Split Box
    # ────────────────────────────────────────────────────────────────────────
    for name, _, _, f1s, color in methods:
        fig.add_trace(go.Box(
            y=f1s, name=name, marker_color=color,
            boxpoints="all", jitter=0.4, pointpos=0,
            line_width=1.5,
            hovertemplate="%{y:.3f}<extra></extra>",
            showlegend=False,
        ), row=1, col=2)
    fig.add_hline(y=0.87, line_dash="dash", line_color=C["paper"], line_width=1,
                  row=1, col=2)

    # ────────────────────────────────────────────────────────────────────────
    # (1,3) HDM Annotations by filter reason
    # ────────────────────────────────────────────────────────────────────────
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
        x=[c for _, c in sorted_r],
        orientation="h",
        marker_color=r_colors[:len(sorted_r)],
        text=[str(c) for _, c in sorted_r], textposition="outside",
        hovertemplate="%{y}: %{x}<extra></extra>",
        showlegend=False,
    ), row=1, col=3)

    # ────────────────────────────────────────────────────────────────────────
    # (2,1) F1 Across Splits (line chart)
    # ────────────────────────────────────────────────────────────────────────
    split_x = [f"Split {i}" for i in range(5)]
    for name, _, _, f1s, color in methods:
        # Shorten name for inline label
        short = name.split("(")[0].strip()
        fig.add_trace(go.Scatter(
            x=split_x, y=f1s, mode="lines+markers+text",
            name=name, line=dict(color=color, width=2.5),
            marker=dict(size=8),
            text=[""] * 4 + [short],  # label at last point
            textposition="middle right", textfont=dict(size=9, color=color),
            hovertemplate="<b>%{x}</b><br>F1: %{y:.3f}<extra></extra>",
            legendgroup=name, showlegend=False,
        ), row=2, col=1)
    fig.add_hline(y=0.87, line_dash="dash", line_color=C["paper"], line_width=1,
                  row=2, col=1)

    # ────────────────────────────────────────────────────────────────────────
    # (2,2) Test Set Composition (stacked via base)
    # ────────────────────────────────────────────────────────────────────────
    g_splits = gemini["splits"]
    split_labels = [f"Split {s['split']}" for s in g_splits]
    n_neg = [s["n_test"] - s["n_pos"] for s in g_splits]
    n_pos = [s["n_pos"] for s in g_splits]
    fig.add_trace(go.Bar(
        x=split_labels, y=n_neg,
        name="Negative", marker_color=C["neg"],
        hovertemplate="<b>%{x}</b><br>Negative: %{y}<extra></extra>",
        legendgroup="comp", showlegend=False,
        offsetgroup="comp",
    ), row=2, col=2)
    fig.add_trace(go.Bar(
        x=split_labels, y=n_pos,
        base=n_neg,  # stack on top of negatives
        name="Positive (HDM)", marker_color=C["pos"],
        hovertemplate="<b>%{x}</b><br>Positive: %{y}<extra></extra>",
        legendgroup="comp", showlegend=False,
        offsetgroup="comp",
    ), row=2, col=2)

    # ────────────────────────────────────────────────────────────────────────
    # (2,3) Predicted vs Actual Positives
    # ────────────────────────────────────────────────────────────────────────
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

    # ────────────────────────────────────────────────────────────────────────
    # (3,1) & (3,2) Confusion Matrices
    # ────────────────────────────────────────────────────────────────────────
    for col_idx, (cm_name, result) in enumerate(
        [("Gemini 3.1 Pro", gemini), ("ASR Hotword", hotword)], start=1
    ):
        tp = fp = fn = tn = 0
        for split in result["splits"]:
            rpt = split["report"]
            n_p = split["n_pos"]
            n_n = split["n_test"] - n_p
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

    # ────────────────────────────────────────────────────────────────────────
    # (3,3) Precision-Recall Curves
    # ────────────────────────────────────────────────────────────────────────
    for split in gemini["splits"]:
        if "precision_curve" in split and "recall_curve" in split:
            fig.add_trace(go.Scatter(
                x=split["recall_curve"], y=split["precision_curve"],
                mode="lines+markers", name=f"Gemini Split {split['split']}",
                line=dict(color=C["gemini"], width=2), opacity=0.6,
                marker=dict(size=5),
                hovertemplate="Recall: %{x:.2f}<br>Precision: %{y:.2f}<extra></extra>",
                showlegend=False,
            ), row=3, col=3)
    fig.add_hline(y=0.091, line_dash="dot", line_color=C["rand50"],
                  annotation_text="Base rate", annotation_font_size=9,
                  row=3, col=3)

    # ────────────────────────────────────────────────────────────────────────
    # (4,1) Gemini Confidence Distribution
    # ────────────────────────────────────────────────────────────────────────
    probs_pos, probs_neg = [], []
    for split in gemini["splits"]:
        probs = split.get("probabilities", [])
        labels = split.get("true_labels", [])
        n_p = split["n_pos"]
        if probs and labels:
            for prob, label in zip(probs, labels):
                (probs_pos if label == 1 else probs_neg).append(prob)
        elif probs:
            probs_pos.extend(probs[:n_p])
            probs_neg.extend(probs[n_p:])

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

    # ────────────────────────────────────────────────────────────────────────
    # (4,2) HDM Duration Distribution
    # ────────────────────────────────────────────────────────────────────────
    durations = [a["duration_ms"] for a in annotations]
    fig.add_trace(go.Histogram(
        x=durations, nbinsx=25, marker_color=C["gemini"], opacity=0.8,
        hovertemplate="Duration: %{x:.0f}ms<br>Count: %{y}<extra></extra>",
        showlegend=False,
    ), row=4, col=2)

    # ────────────────────────────────────────────────────────────────────────
    # (4,3) HDMs by Speaker (pie)
    # ────────────────────────────────────────────────────────────────────────
    speakers = Counter(a["speaker"] for a in annotations)
    fig.add_trace(go.Pie(
        labels=list(speakers.keys()),
        values=list(speakers.values()),
        hole=0.4,
        hovertemplate="Speaker %{label}<br>%{value} HDMs (%{percent})<extra></extra>",
        textinfo="label+value",
        showlegend=False,
    ), row=4, col=3)

    # ── Layout ──────────────────────────────────────────────────────────────
    fig.update_layout(
        title=dict(
            text="<b>HDM Detection — Trial Results Dashboard</b>",
            x=0.5, y=0.995, font=dict(size=22, color="#1565C0"),
        ),
        height=2600,
        width=1600,
        template="plotly_white",
        font=dict(family="Inter, system-ui, sans-serif", size=11, color=C["text"]),
        showlegend=False,
        barmode="group",
        margin=dict(t=120, b=40, l=60, r=40),
    )

    # Enable automargin on all axes so labels are never clipped
    fig.update_xaxes(automargin=True, tickfont=dict(size=10))
    fig.update_yaxes(automargin=True, tickfont=dict(size=10))

    # Per-subplot axis labels
    fig.update_yaxes(title_text="F1 Score", range=[0, 1.08], row=1, col=1)
    fig.update_xaxes(tickangle=-30, row=1, col=1)  # angle bar labels
    fig.update_yaxes(range=[-0.05, 1.05], row=1, col=2)
    fig.update_xaxes(tickangle=-30, row=1, col=2)
    fig.update_xaxes(title_text="Count", row=1, col=3)
    fig.update_yaxes(title_text="F1 Score", range=[0, 1.05], row=2, col=1)
    fig.update_yaxes(title_text="Segments", row=2, col=2)
    fig.update_yaxes(title_text="Count", row=2, col=3)
    fig.update_xaxes(tickangle=-30, row=2, col=3)  # angle pred vs actual labels
    fig.update_xaxes(title_text="Recall", range=[-0.05, 1.05], row=3, col=3)
    fig.update_yaxes(title_text="Precision", range=[-0.05, 1.05], row=3, col=3)
    fig.update_xaxes(title_text="Predicted Probability", row=4, col=1)
    fig.update_yaxes(title_text="Count", row=4, col=1)
    fig.update_xaxes(title_text="Duration (ms)", row=4, col=2)
    fig.update_yaxes(title_text="Count", row=4, col=2)

    # ── Write HTML ──────────────────────────────────────────────────────────
    plotly_config = {
        "displayModeBar": True,
        "modeBarButtonsToAdd": ["toggleSpikelines"],
        "toImageButtonOptions": {"format": "png", "scale": 2},
    }

    # Get the chart div (not full HTML — we'll wrap it ourselves)
    chart_html = fig.to_html(
        include_plotlyjs=False, full_html=False, config=plotly_config,
    )

    page_html = _build_full_page(chart_html, gemini, hotword, random_bl, annotations)

    # Main output (self-contained, works offline)
    output = RESULTS_DIR / "dashboard.html"
    # Embed plotly.js inline for offline use
    plotly_js_cdn = "https://cdn.plot.ly/plotly-3.0.1.min.js"
    output.write_text(page_html.replace("{{PLOTLY_SRC}}", plotly_js_cdn))
    print(f"Dashboard saved to {output}")

    # GitHub Pages version (same, CDN-loaded)
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    gh_output = docs_dir / "index.html"
    gh_output.write_text(page_html.replace("{{PLOTLY_SRC}}", plotly_js_cdn))
    print(f"GitHub Pages version saved to {gh_output}")
    print(f"\nLocal preview: file://{output.resolve()}")
    return output


def _build_full_page(chart_html, gemini, hotword, random_bl, annotations):
    """Build the complete HTML page with explanatory content around the chart."""

    # Compute summary stats for the info cards
    g_precision_sum = sum(
        s["report"]["1"]["precision"] for s in gemini["splits"]
    )
    g_recall_sum = sum(
        s["report"]["1"]["recall"] for s in gemini["splits"]
    )
    g_prec = g_precision_sum / len(gemini["splits"])
    g_rec = g_recall_sum / len(gemini["splits"])

    h_prec_sum = sum(
        s["report"]["1"]["precision"] for s in hotword["splits"]
    )
    h_rec_sum = sum(
        s["report"]["1"]["recall"] for s in hotword["splits"]
    )
    h_prec = h_prec_sum / len(hotword["splits"])
    h_rec = h_rec_sum / len(hotword["splits"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>HDM Detection — Trial Results Dashboard</title>
<script src="{{{{PLOTLY_SRC}}}}"></script>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{
    font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: #f5f7fa; color: #1a1a2e; line-height: 1.6;
  }}
  .header {{
    background: linear-gradient(135deg, #1565C0 0%, #0D47A1 100%);
    color: white; padding: 48px 24px 40px; text-align: center;
  }}
  .header h1 {{ font-size: 2rem; font-weight: 700; margin-bottom: 8px; }}
  .header p {{ font-size: 1rem; opacity: 0.85; max-width: 700px; margin: 0 auto; }}
  .container {{ max-width: 1600px; margin: 0 auto; padding: 0 24px; }}

  /* KPI cards */
  .kpi-row {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 16px; margin: -32px auto 32px; position: relative; z-index: 1;
  }}
  .kpi-card {{
    background: white; border-radius: 12px; padding: 20px 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.08); text-align: center;
    border-top: 4px solid #ccc;
  }}
  .kpi-card .label {{ font-size: 0.8rem; color: #666; text-transform: uppercase;
    letter-spacing: 0.05em; margin-bottom: 4px; }}
  .kpi-card .value {{ font-size: 1.8rem; font-weight: 700; }}
  .kpi-card .sub {{ font-size: 0.75rem; color: #999; margin-top: 2px; }}

  /* Section blocks */
  .section {{
    background: white; border-radius: 12px; padding: 32px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 28px;
  }}
  .section h2 {{
    font-size: 1.3rem; color: #1565C0; margin-bottom: 16px;
    border-bottom: 2px solid #e3f2fd; padding-bottom: 8px;
  }}
  .section h3 {{ font-size: 1rem; color: #333; margin: 16px 0 8px; }}
  .section p {{ color: #444; margin-bottom: 12px; font-size: 0.92rem; }}

  /* Metric definitions grid */
  .metrics-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 16px; margin: 16px 0;
  }}
  .metric-card {{
    background: #f8f9fa; border-radius: 8px; padding: 16px 20px;
    border-left: 4px solid #1565C0;
  }}
  .metric-card.green {{ border-left-color: #66BB6A; }}
  .metric-card.orange {{ border-left-color: #FF6D00; }}
  .metric-card.red {{ border-left-color: #EF5350; }}
  .metric-card .metric-name {{ font-weight: 700; font-size: 0.95rem; color: #222; }}
  .metric-card .metric-desc {{ font-size: 0.83rem; color: #555; margin-top: 4px; }}
  .metric-card .metric-example {{ font-size: 0.78rem; color: #888; margin-top: 6px;
    font-style: italic; }}

  /* Method cards */
  .methods-grid {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 16px; margin: 16px 0;
  }}
  .method-card {{
    background: #f8f9fa; border-radius: 8px; padding: 20px;
    border-top: 4px solid #ccc; position: relative;
  }}
  .method-card .method-name {{ font-weight: 700; font-size: 1rem; margin-bottom: 4px; }}
  .method-card .method-f1 {{ font-size: 1.4rem; font-weight: 700; margin: 8px 0; }}
  .method-card .method-desc {{ font-size: 0.83rem; color: #555; }}
  .method-card .method-detail {{ font-size: 0.78rem; color: #777; margin-top: 8px; }}

  /* Chart panel descriptions */
  .chart-guide {{
    display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 12px; margin: 16px 0;
  }}
  .chart-guide-item {{
    background: #f8f9fa; border-radius: 6px; padding: 12px 16px;
  }}
  .chart-guide-item .chart-name {{
    font-weight: 600; font-size: 0.85rem; color: #1565C0; margin-bottom: 4px;
  }}
  .chart-guide-item .chart-desc {{ font-size: 0.8rem; color: #555; }}

  /* Chart container */
  .chart-section {{ margin: 28px 0; }}
  .chart-section .plotly-chart {{ width: 100%; overflow-x: auto; }}

  .footer {{
    text-align: center; padding: 32px 24px; color: #999; font-size: 0.8rem;
  }}
  .footer a {{ color: #1565C0; }}

  /* Collapsible */
  details {{ margin: 12px 0; }}
  details summary {{
    cursor: pointer; font-weight: 600; font-size: 0.9rem; color: #1565C0;
    padding: 8px 0;
  }}
  details[open] summary {{ margin-bottom: 8px; }}
</style>
</head>
<body>

<!-- ═══ HEADER ═══ -->
<div class="header">
  <h1>HDM Detection — Trial Results Dashboard</h1>
  <p>Replication of Collins et al. (2025) on the AMI Meeting Corpus.
     Comparing 4 methods for detecting Hearing Difficulty Moments using
     5-Fold Monte Carlo Cross-Validation. All charts are interactive —
     hover, zoom, and pan to explore.</p>
</div>

<div class="container">

<!-- ═══ KPI CARDS ═══ -->
<div class="kpi-row">
  <div class="kpi-card" style="border-top-color: #4285F4;">
    <div class="label">Gemini 3.1 Pro</div>
    <div class="value" style="color: #4285F4;">{gemini['avg_f1']:.3f}</div>
    <div class="sub">F1 &plusmn; {gemini['std_f1']:.3f} &middot; Best method</div>
  </div>
  <div class="kpi-card" style="border-top-color: #FF6D00;">
    <div class="label">ASR Hotword</div>
    <div class="value" style="color: #FF6D00;">{hotword['avg_f1']:.3f}</div>
    <div class="sub">F1 &plusmn; {hotword['std_f1']:.3f} &middot; Whisper + keywords</div>
  </div>
  <div class="kpi-card" style="border-top-color: #9E9E9E;">
    <div class="label">Random 50/50</div>
    <div class="value" style="color: #9E9E9E;">{random_bl['random_50_50']['avg_f1']:.3f}</div>
    <div class="sub">F1 &middot; Coin-flip baseline</div>
  </div>
  <div class="kpi-card" style="border-top-color: #BDBDBD;">
    <div class="label">Random Base-rate</div>
    <div class="value" style="color: #999;">{random_bl['random_base_rate']['avg_f1']:.3f}</div>
    <div class="sub">F1 &middot; 9.1% positive rate</div>
  </div>
  <div class="kpi-card" style="border-top-color: #E91E63;">
    <div class="label">Collins et al. (Paper)</div>
    <div class="value" style="color: #E91E63;">0.870</div>
    <div class="sub">Gemini 1.5 Pro &middot; SWDA/MRDA</div>
  </div>
</div>

<!-- ═══ WHAT ARE HDMs? ═══ -->
<div class="section">
  <h2>What are Hearing Difficulty Moments (HDMs)?</h2>
  <p>An HDM occurs when a listener in a conversation struggles to understand what was said.
     They typically respond with short phrases like <strong>"What?"</strong>, <strong>"Huh?"</strong>,
     <strong>"Sorry?"</strong>, or <strong>"Can you repeat that?"</strong>. Detecting these moments
     automatically from audio could help improve hearing aids, meeting transcription tools,
     and accessibility systems.</p>
  <p>This project tests whether AI models can detect HDMs from 4-second audio clips of meeting
     recordings. We compare four methods — from random guessing (the floor) to Google's Gemini
     AI model (the best performer). The dataset contains <strong>1,155 audio segments</strong>
     from the AMI Meeting Corpus, with only <strong>9.1%</strong> being actual HDMs (105 out of 1,155),
     making this a challenging imbalanced classification task.</p>
</div>

<!-- ═══ KEY METRICS ═══ -->
<div class="section">
  <h2>Understanding the Metrics</h2>
  <p>These are the key measurements used to evaluate how well each method detects HDMs.
     All metrics range from 0 (worst) to 1 (perfect).</p>
  <div class="metrics-grid">
    <div class="metric-card">
      <div class="metric-name">F1 Score</div>
      <div class="metric-desc">The primary metric. It balances precision and recall into a
        single number. Formally, F1 = 2 &times; (Precision &times; Recall) / (Precision + Recall).
        A high F1 means the model is good at <em>both</em> finding real HDMs and avoiding false alarms.</div>
      <div class="metric-example">Example: Gemini's F1 of 0.58 means it does a reasonable job but still
        makes significant errors on both sides.</div>
    </div>
    <div class="metric-card green">
      <div class="metric-name">Precision</div>
      <div class="metric-desc">Of all the moments the model flagged as HDMs, what fraction actually were?
        High precision = few false alarms. Low precision = the model cries wolf too often.</div>
      <div class="metric-example">Gemini precision: {g_prec:.2f} — about {int(g_prec*100)}% of its "HDM" predictions
        are correct. Hotword precision: {h_prec:.2f}.</div>
    </div>
    <div class="metric-card orange">
      <div class="metric-name">Recall (Sensitivity)</div>
      <div class="metric-desc">Of all the real HDMs in the data, what fraction did the model catch?
        High recall = few missed HDMs. Low recall = many real HDMs go undetected.</div>
      <div class="metric-example">Gemini recall: {g_rec:.2f} — catches {int(g_rec*100)}% of real HDMs.
        Hotword recall: {h_rec:.2f}.</div>
    </div>
    <div class="metric-card red">
      <div class="metric-name">Confusion Matrix</div>
      <div class="metric-desc">A 2&times;2 table showing the four possible outcomes: True Positives
        (correctly detected HDMs), True Negatives (correctly ignored non-HDMs), False Positives
        (false alarms), and False Negatives (missed HDMs). The diagonal = correct predictions.</div>
      <div class="metric-example">Gemini's main weakness: too many False Positives (134 false alarms).
        Hotword's: too many False Negatives (39 missed HDMs).</div>
    </div>
  </div>

  <details>
    <summary>More metrics explained...</summary>
    <div class="metrics-grid" style="margin-top: 8px;">
      <div class="metric-card">
        <div class="metric-name">Precision-Recall Curve</div>
        <div class="metric-desc">Shows the trade-off between precision and recall as the model's
          confidence threshold changes. A curve that stays high and to the right is better.
          The dotted "base rate" line (9.1%) is what random guessing achieves.</div>
      </div>
      <div class="metric-card">
        <div class="metric-name">Monte Carlo Cross-Validation (MCCV)</div>
        <div class="metric-desc">Instead of testing once, we randomly split the data 5 times into
          training (80%) and test (20%) sets. Each split uses different meetings for training and
          testing. This gives 5 independent F1 scores, showing how stable the method is.</div>
      </div>
      <div class="metric-card">
        <div class="metric-name">Standard Deviation (&plusmn;)</div>
        <div class="metric-desc">Measures how much the F1 score varies across the 5 splits. A small
          &plusmn; means consistent performance. Gemini's &plusmn;0.09 means its F1 can swing by about
          0.09 in either direction depending on which meetings are in the test set.</div>
      </div>
      <div class="metric-card">
        <div class="metric-name">Class Imbalance</div>
        <div class="metric-desc">Only 9.1% of segments are HDMs. This makes accuracy misleading —
          a model that always predicts "not HDM" gets 91% accuracy but catches zero real HDMs (F1 = 0).
          That's why we use F1 instead of accuracy.</div>
      </div>
    </div>
  </details>
</div>

<!-- ═══ METHODS ═══ -->
<div class="section">
  <h2>Detection Methods Compared</h2>
  <p>We tested four methods of increasing sophistication. The random baselines establish what
     you'd get with no real detection ability. The hotword approach adds basic speech recognition.
     Gemini brings full audio understanding with AI reasoning.</p>
  <div class="methods-grid">
    <div class="method-card" style="border-top-color: #9E9E9E;">
      <div class="method-name">Random 50/50 Baseline</div>
      <div class="method-f1" style="color: #9E9E9E;">F1 = 0.154</div>
      <div class="method-desc">Flips a fair coin for each audio segment — 50% chance of predicting
        "HDM" regardless of what's in the audio. Never listens to anything.</div>
      <div class="method-detail"><strong>Why so low:</strong> The dataset is 91% negative. Guessing
        positive half the time creates massive false positives, tanking precision to ~0.09.
        Averaged over 100 random seeds per split.</div>
    </div>
    <div class="method-card" style="border-top-color: #BDBDBD;">
      <div class="method-name">Random Base-rate Baseline</div>
      <div class="method-f1" style="color: #999;">F1 = 0.090</div>
      <div class="method-desc">A weighted coin that predicts "HDM" only 9.1% of the time, matching
        the actual HDM rate. Tests whether knowing the base rate helps.</div>
      <div class="method-detail"><strong>Why even lower:</strong> It rarely predicts positive, so it
        catches almost no real HDMs (near-zero recall). Knowing the base rate alone isn't useful
        for detection.</div>
    </div>
    <div class="method-card" style="border-top-color: #FF6D00;">
      <div class="method-name">ASR Hotword Heuristic</div>
      <div class="method-f1" style="color: #FF6D00;">F1 = 0.226</div>
      <div class="method-desc">Transcribes each 4s audio clip with OpenAI Whisper, then checks
        for keywords like "huh", "what", "pardon", "sorry", "repeat that".</div>
      <div class="method-detail"><strong>Limitation:</strong> Depends on clean transcription.
        In AMI meetings with 4 overlapping speakers, Whisper often can't hear the "What?" clearly
        enough to transcribe it. Ignores all acoustic cues (tone, pitch, hesitation).</div>
    </div>
    <div class="method-card" style="border-top-color: #4285F4;">
      <div class="method-name">Gemini 3.1 Pro (10-shot)</div>
      <div class="method-f1" style="color: #4285F4;">F1 = 0.583</div>
      <div class="method-desc">Google's Gemini AI listens to the raw audio and decides if someone
        is having hearing difficulty. Shown 10 labelled examples first (5 HDM + 5 non-HDM) to
        learn the pattern.</div>
      <div class="method-detail"><strong>Key advantage:</strong> Processes raw audio waveforms,
        capturing tone, pitch, hesitation, and the Lombard effect — acoustic signals invisible to
        keyword-based approaches. <strong>Weakness:</strong> tends to over-predict
        (high recall, lower precision).</div>
    </div>
  </div>
</div>

<!-- ═══ CHART GUIDE ═══ -->
<div class="section">
  <h2>Reading the Charts</h2>
  <p>Each panel below shows a different aspect of the results. Here's what to look for:</p>
  <div class="chart-guide">
    <div class="chart-guide-item">
      <div class="chart-name">F1 Score Comparison</div>
      <div class="chart-desc">Bar chart of average F1 for each method. Taller = better.
        Error bars show variability. The pink dashed line is the paper's benchmark (0.87).</div>
    </div>
    <div class="chart-guide-item">
      <div class="chart-name">Per-Split F1 Distribution</div>
      <div class="chart-desc">Box plots showing how F1 varies across the 5 test splits.
        Wide box = inconsistent. Each dot is one split's F1.</div>
    </div>
    <div class="chart-guide-item">
      <div class="chart-name">HDM Annotations by Type</div>
      <div class="chart-desc">How the 149 ground-truth HDMs were classified: strong keywords
        ("What?", "Huh?"), explicit non-understanding, or short questions.</div>
    </div>
    <div class="chart-guide-item">
      <div class="chart-name">F1 Across CV Splits</div>
      <div class="chart-desc">Line chart tracking each method across all 5 splits. Flat lines
        = stable. Gemini's ups and downs show sensitivity to which meetings are tested.</div>
    </div>
    <div class="chart-guide-item">
      <div class="chart-name">Test Set Composition</div>
      <div class="chart-desc">How many positive (HDM, red) vs negative (green) segments per
        split. Shows the heavy ~10:1 class imbalance that makes this task hard.</div>
    </div>
    <div class="chart-guide-item">
      <div class="chart-name">Predicted vs Actual Positives</div>
      <div class="chart-desc">Faded bars = actual HDM count. Solid bars = model's predictions.
        Gemini over-predicts (solid > faded). Hotword under-predicts (solid &lt; faded).</div>
    </div>
    <div class="chart-guide-item">
      <div class="chart-name">Confusion Matrices</div>
      <div class="chart-desc">2&times;2 grids: top-left (True Neg) and bottom-right (True Pos) are
        correct. Top-right = false alarms. Bottom-left = missed HDMs. Darker = more.</div>
    </div>
    <div class="chart-guide-item">
      <div class="chart-name">Precision-Recall Curves</div>
      <div class="chart-desc">Shows how precision and recall trade off at different confidence
        levels. Higher and further right = better. Each line is one CV split.</div>
    </div>
    <div class="chart-guide-item">
      <div class="chart-name">Confidence Distribution</div>
      <div class="chart-desc">How confident Gemini is in its predictions. Blue = negative
        predictions, red = positive. The model is very binary — mostly 0.1 or 0.9, rarely
        in between.</div>
    </div>
    <div class="chart-guide-item">
      <div class="chart-name">HDM Duration Distribution</div>
      <div class="chart-desc">How long each HDM utterance lasts in milliseconds. Most are short
        bursts under 1 second — quick "What?" or "Huh?" responses.</div>
    </div>
    <div class="chart-guide-item">
      <div class="chart-name">HDMs by Speaker</div>
      <div class="chart-desc">Which meeting participants (A, B, C, D) had the most hearing
        difficulty moments. Shows whether HDMs cluster around certain speakers.</div>
    </div>
  </div>
</div>

<!-- ═══ CHARTS ═══ -->
<div class="chart-section">
  <div class="plotly-chart">
    {chart_html}
  </div>
</div>

<!-- ═══ INTERPRETATION ═══ -->
<div class="section">
  <h2>Key Takeaways</h2>
  <h3>1. Audio AI works, but the task is harder on meeting data</h3>
  <p>Gemini 3.1 Pro (F1 = 0.58) substantially outperforms all baselines, confirming that audio
     language models can detect HDMs. However, it falls short of the paper's 0.87, mainly because
     AMI meeting audio has 4 speakers mixed into one channel with overlapping speech and background
     noise — much harder than the paper's cleaner telephone conversations.</p>

  <h3>2. Keywords alone aren't enough</h3>
  <p>The ASR Hotword approach (F1 = 0.23) shows that simply looking for words like "What?" in
     a transcript is unreliable. In noisy meetings, the speech-to-text model often can't transcribe
     short, quiet HDM utterances like "Huh?" accurately enough for keyword matching to work.</p>

  <h3>3. Gemini over-predicts but catches most real HDMs</h3>
  <p>The confusion matrices reveal Gemini's strategy: it predicts positive aggressively (229
     predicted vs 105 actual), catching 95 of 105 real HDMs (90% recall) but at the cost of
     134 false alarms. The Hotword baseline has the opposite problem — it's too conservative,
     missing 39 of 48 real HDMs.</p>

  <h3>4. Results vary significantly by data split</h3>
  <p>Gemini's per-split F1 ranges from 0.47 to 0.70 — a wide spread. This is because with only
     15-26 positive examples per test split, a handful of misclassifications can swing the F1
     dramatically. More data would likely produce more stable estimates.</p>
</div>

</div><!-- /container -->

<div class="footer">
  <p>Data from replication of Collins et al. (2025)
    <em>"Identifying Hearing Difficulty Moments in Conversational Audio"</em>
    on the AMI Meeting Corpus.</p>
  <p style="margin-top: 8px;">
    <a href="https://github.com/chozillla/CollinsPaper">GitHub Repository</a> &middot;
    <a href="https://arxiv.org/abs/2507.23590">Original Paper</a>
  </p>
</div>

</body>
</html>"""


if __name__ == "__main__":
    build_dashboard()
