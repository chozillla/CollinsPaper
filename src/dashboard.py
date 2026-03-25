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

    # Main output (self-contained, works offline)
    output = RESULTS_DIR / "dashboard.html"
    fig.write_html(str(output), include_plotlyjs=True, full_html=True,
                   config=plotly_config)
    print(f"Dashboard saved to {output}")

    # GitHub Pages version (CDN-loaded, much smaller file)
    docs_dir = ROOT / "docs"
    docs_dir.mkdir(exist_ok=True)
    gh_output = docs_dir / "index.html"
    fig.write_html(str(gh_output), include_plotlyjs="cdn", full_html=True,
                   config=plotly_config)
    print(f"GitHub Pages version saved to {gh_output}")
    print(f"\nTo enable GitHub Pages: Settings > Pages > Source: 'Deploy from a branch' > Branch: main /docs")
    print(f"Local preview: file://{output.resolve()}")
    return output


if __name__ == "__main__":
    build_dashboard()
