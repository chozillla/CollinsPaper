"""
Validation Dashboard — explore v4 classifier predictions with audio playback.

Browse test set predictions by split, filter by category (TP/FP/FN/TN),
listen to audio clips, and verify P/N labels. Also browse the few-shot
examples (P and N) used for each split.

Usage:
    python src/validation_dashboard.py
    Then open http://localhost:8766 in your browser.
"""

import gc
import json
import io
import base64
import math
import numpy as np
import soundfile as sf
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from collections import defaultdict

AUDIO_DIR = Path("data/audio")
DATASET_DIR = Path("data/dataset")
RESULTS_FILE = Path("results/gpt4o_20shot_v4_results.json")
HUMAN_LABELS_FILE = Path("data/hdm_labels.json")
SAMPLE_RATE = 16000
SEGMENT_DURATION = 4.0
CONTEXT_BEFORE = 4.0
CONTEXT_AFTER = 4.0
PORT = 8766

RANDOM_SEED = 42
N_POS_SHOTS = 12
N_NEG_SHOTS = 8

# Global state
split_data = []
shot_data = []  # per-split few-shot examples


def load_data():
    """Load results + dataset metadata, build per-split test items and shots."""
    global split_data, shot_data

    with open(RESULTS_FILE) as f:
        results = json.load(f)

    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    human_labels = {}
    if HUMAN_LABELS_FILE.exists():
        with open(HUMAN_LABELS_FILE) as f:
            human_labels = json.load(f)

    all_examples = meta["positive"] + meta["negative"]
    n_positives = 149  # matches v4 classifier constant

    for split_idx, split_result in enumerate(results["splits"]):
        test_meetings = set(meta["splits"][split_idx]["test"])
        train_meetings = set(meta["splits"][split_idx]["train"])

        # --- Build test items ---
        test_items = []
        for i, ex in enumerate(all_examples):
            if ex["meeting_id"] in test_meetings:
                test_items.append({"global_idx": i, "meta": ex})

        predictions = split_result["predictions"]
        true_labels = split_result["true_labels"]
        probabilities = split_result["probabilities"]

        items = []
        for j, item in enumerate(test_items):
            if j >= len(predictions):
                break
            pred = predictions[j]
            true = true_labels[j]
            prob = probabilities[j]

            if pred == 1 and true == 1:
                category = "TP"
            elif pred == 1 and true == 0:
                category = "FP"
            elif pred == 0 and true == 1:
                category = "FN"
            else:
                category = "TN"

            items.append({
                "idx": j,
                "global_idx": item["global_idx"],
                "meeting_id": item["meta"]["meeting_id"],
                "speaker": item["meta"].get("speaker", "?"),
                "sample_time": item["meta"]["sample_time"],
                "text": item["meta"].get("text", ""),
                "true_label": true,
                "prediction": pred,
                "probability": prob,
                "category": category,
            })

        split_data.append({
            "split": split_idx,
            "f1": split_result["f1"],
            "items": items,
            "n_test": len(items),
            "n_tp": sum(1 for it in items if it["category"] == "TP"),
            "n_fp": sum(1 for it in items if it["category"] == "FP"),
            "n_fn": sum(1 for it in items if it["category"] == "FN"),
            "n_tn": sum(1 for it in items if it["category"] == "TN"),
        })

        # --- Build few-shot examples (replicate v4 selection logic) ---
        train_indices = [i for i, ex in enumerate(all_examples)
                         if ex["meeting_id"] in train_meetings]

        verified_pos = []
        hard_neg = []
        unverified_pos = []
        random_neg = []

        for idx in train_indices:
            if idx < n_positives:
                human = human_labels.get(str(idx))
                if human == "yes":
                    verified_pos.append(idx)
                elif human == "no":
                    hard_neg.append(idx)
                else:
                    unverified_pos.append(idx)
            else:
                random_neg.append(idx)

        np.random.seed(RANDOM_SEED + split_idx)
        pos_pool = verified_pos if len(verified_pos) >= N_POS_SHOTS else verified_pos + unverified_pos
        selected_pos = list(np.random.choice(pos_pool, size=min(N_POS_SHOTS, len(pos_pool)), replace=False))

        n_hard = min(len(hard_neg), N_NEG_SHOTS // 2)
        n_random = N_NEG_SHOTS - n_hard
        hard_selected = list(np.random.choice(hard_neg, size=n_hard, replace=False)) if hard_neg else []
        random_selected = list(np.random.choice(random_neg, size=min(n_random, len(random_neg)), replace=False))
        selected_neg = hard_selected + random_selected

        # Build interleaved order (same as classifier: P,P,N,P,P,N...)
        shots = []
        pi, ni = 0, 0
        while pi < len(selected_pos) or ni < len(selected_neg):
            for _ in range(2):
                if pi < len(selected_pos):
                    idx = int(selected_pos[pi])
                    ex = all_examples[idx]
                    source = "human-verified" if human_labels.get(str(idx)) == "yes" else "unverified"
                    shots.append({
                        "idx": idx,
                        "meeting_id": ex["meeting_id"],
                        "speaker": ex.get("speaker", "?"),
                        "sample_time": ex["sample_time"],
                        "text": ex.get("text", ""),
                        "label": 1,
                        "shot_type": "positive",
                        "source": source,
                    })
                    pi += 1
            if ni < len(selected_neg):
                idx = int(selected_neg[ni])
                ex = all_examples[idx]
                is_hard = idx in hard_neg
                source = "hard-negative (human-rejected)" if is_hard else "random-negative"
                shots.append({
                    "idx": idx,
                    "meeting_id": ex["meeting_id"],
                    "speaker": ex.get("speaker", "?"),
                    "sample_time": ex["sample_time"],
                    "text": ex.get("text", ""),
                    "label": 0,
                    "shot_type": "negative",
                    "source": source,
                })
                ni += 1

        shot_data.append({
            "split": split_idx,
            "shots": shots,
            "n_pos": len(selected_pos),
            "n_neg": len(selected_neg),
            "n_hard_neg": len(hard_selected),
            "n_random_neg": len(random_selected),
        })

    print(f"Loaded {len(split_data)} splits")
    for i, sd in enumerate(split_data):
        sh = shot_data[i]
        print(f"  Split {sd['split']+1}: {sd['n_test']} test items, "
              f"F1={sd['f1']:.4f}, TP={sd['n_tp']} FP={sd['n_fp']} FN={sd['n_fn']} TN={sd['n_tn']}, "
              f"shots={sh['n_pos']}P+{sh['n_neg']}N ({sh['n_hard_neg']} hard neg)")


def get_audio_b64(meeting_id, sample_time):
    """Extract and return base64-encoded audio clips for a segment."""
    audio_path = AUDIO_DIR / f"{meeting_id}.Mix-Headset.wav"
    if not audio_path.exists():
        return None

    audio, sr = sf.read(str(audio_path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    hdm_end = int(sample_time * sr)
    hdm_start = int((sample_time - SEGMENT_DURATION) * sr)
    ctx_start = int((sample_time - SEGMENT_DURATION - CONTEXT_BEFORE) * sr)
    ext_end = int((sample_time + CONTEXT_AFTER) * sr)

    ctx_start = max(0, ctx_start)
    hdm_start = max(0, hdm_start)
    hdm_end = min(len(audio), hdm_end)
    ext_end = min(len(audio), ext_end)

    def to_b64(arr):
        buf = io.BytesIO()
        sf.write(buf, arr.astype(np.float32), sr, format="WAV")
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")

    result = {
        "hdm": to_b64(audio[hdm_start:hdm_end]),
        "context": to_b64(audio[ctx_start:hdm_start]),
        "extended": to_b64(audio[ctx_start:ext_end]),
    }

    del audio
    gc.collect()
    return result


HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>HDM Validation Dashboard</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f1a; color: #eee; }

  .top-bar { background: #16213e; padding: 14px 24px; border-bottom: 1px solid #1a3a5c; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }
  .top-bar h1 { font-size: 18px; color: #e94560; white-space: nowrap; }

  .controls { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
  .controls label { font-size: 13px; color: #888; }
  .controls select { background: #0f3460; color: #eee; border: 1px solid #1a4a8a; padding: 6px 12px; border-radius: 6px; font-size: 13px; cursor: pointer; }
  .controls select:hover { border-color: #e94560; }

  .tab-bar { display: flex; gap: 0; background: #111; border-bottom: 1px solid #1a1a2e; }
  .tab { padding: 10px 24px; font-size: 13px; font-weight: 600; cursor: pointer; border-bottom: 3px solid transparent; color: #888; transition: all 0.15s; }
  .tab:hover { color: #ccc; background: #1a1a2e; }
  .tab.active { color: #e94560; border-bottom-color: #e94560; }

  .filter-chips { display: flex; gap: 6px; }
  .chip { padding: 4px 12px; border-radius: 20px; font-size: 11px; font-weight: 600; cursor: pointer; border: 2px solid transparent; transition: all 0.15s; }
  .chip.active { border-color: #fff; }
  .chip-all { background: #333; }
  .chip-tp { background: #1b5e20; }
  .chip-fp { background: #b71c1c; }
  .chip-fn { background: #e65100; }
  .chip-tn { background: #1a3a5c; }
  .chip-pos { background: #1b5e20; }
  .chip-neg { background: #b71c1c; }

  .split-stats { display: flex; gap: 14px; padding: 10px 24px; background: #0a0a14; font-size: 12px; color: #aaa; border-bottom: 1px solid #1a1a2e; flex-wrap: wrap; }
  .stat { display: flex; gap: 4px; }
  .stat b { color: #eee; }

  .main { display: flex; height: calc(100vh - 180px); }

  .sidebar { width: 320px; min-width: 320px; background: #111; overflow-y: auto; border-right: 1px solid #1a1a2e; }
  .sidebar-item { padding: 10px 16px; border-bottom: 1px solid #1a1a2e; cursor: pointer; transition: background 0.1s; display: flex; gap: 10px; align-items: center; }
  .sidebar-item:hover { background: #1a1a2e; }
  .sidebar-item.active { background: #16213e; border-left: 3px solid #e94560; }
  .sidebar-badge { font-size: 10px; font-weight: 700; padding: 2px 8px; border-radius: 10px; flex-shrink: 0; white-space: nowrap; }
  .badge-tp { background: #1b5e20; color: #a5d6a7; }
  .badge-fp { background: #b71c1c; color: #ef9a9a; }
  .badge-fn { background: #e65100; color: #ffcc80; }
  .badge-tn { background: #1a3a5c; color: #90caf9; }
  .badge-pos { background: #1b5e20; color: #a5d6a7; }
  .badge-neg { background: #b71c1c; color: #ef9a9a; }
  .sidebar-info { flex: 1; min-width: 0; }
  .sidebar-meeting { font-size: 13px; font-weight: 600; }
  .sidebar-detail { font-size: 11px; color: #888; margin-top: 2px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .sidebar-conf { font-size: 11px; color: #666; }

  .detail-panel { flex: 1; overflow-y: auto; padding: 24px; }
  .detail-empty { display: flex; align-items: center; justify-content: center; height: 100%; color: #555; font-size: 16px; }

  .detail-card { background: #16213e; border-radius: 12px; padding: 24px; max-width: 750px; }
  .detail-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; }
  .detail-title { font-size: 20px; font-weight: 700; }
  .detail-category { font-size: 14px; font-weight: 700; padding: 4px 16px; border-radius: 20px; }

  .meta-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; margin-bottom: 20px; }
  .meta-box { background: #0f3460; padding: 10px 14px; border-radius: 8px; }
  .meta-label { font-size: 11px; color: #888; text-transform: uppercase; margin-bottom: 2px; }
  .meta-value { font-size: 15px; font-weight: 600; }

  .label-row { display: flex; gap: 20px; margin-bottom: 20px; }
  .label-box { flex: 1; padding: 14px; border-radius: 8px; text-align: center; background: #0f3460; }
  .label-box .label-title { font-size: 11px; color: #888; text-transform: uppercase; }
  .label-box .label-val { font-size: 28px; font-weight: 800; margin-top: 4px; }
  .label-p { color: #4caf50; }
  .label-n { color: #f44336; }

  .confidence-bar { margin-bottom: 20px; }
  .conf-label { font-size: 12px; color: #888; margin-bottom: 6px; }
  .conf-track { background: #0a1628; height: 24px; border-radius: 12px; overflow: hidden; position: relative; }
  .conf-fill { height: 100%; border-radius: 12px; transition: width 0.3s; display: flex; align-items: center; justify-content: flex-end; padding-right: 8px; font-size: 11px; font-weight: 700; }
  .conf-fill.high { background: linear-gradient(90deg, #1b5e20, #4caf50); }
  .conf-fill.low { background: linear-gradient(90deg, #b71c1c, #f44336); }
  .conf-fill.mid { background: linear-gradient(90deg, #e65100, #ff9800); }

  .transcript-box { background: #0a1628; padding: 18px; border-radius: 8px; margin-bottom: 20px; text-align: center; font-size: 20px; font-style: italic; min-height: 60px; display: flex; align-items: center; justify-content: center; color: #ccc; }
  .no-transcript { color: #555; font-size: 14px; font-style: normal; }

  .source-tag { display: inline-block; font-size: 11px; padding: 3px 10px; border-radius: 12px; margin-bottom: 16px; }
  .source-verified { background: #1b5e20; color: #a5d6a7; }
  .source-unverified { background: #555; color: #ccc; }
  .source-hard { background: #e65100; color: #ffcc80; }
  .source-random { background: #1a3a5c; color: #90caf9; }

  .audio-section { margin-bottom: 14px; }
  .audio-title { font-size: 13px; color: #e94560; font-weight: 600; margin-bottom: 4px; }
  .audio-desc { font-size: 11px; color: #666; margin-bottom: 6px; }
  audio { width: 100%; height: 36px; border-radius: 8px; }

  .nav-buttons { display: flex; gap: 12px; margin-top: 20px; }
  .nav-btn { flex: 1; padding: 12px; background: #0f3460; border: none; color: #eee; border-radius: 8px; font-size: 14px; cursor: pointer; font-weight: 600; }
  .nav-btn:hover { background: #1a4a8a; }
  .nav-btn:disabled { opacity: 0.3; cursor: default; }

  .keyboard-hint { text-align: center; color: #555; font-size: 11px; margin-top: 12px; }

  .shot-order { font-size: 12px; color: #666; margin-bottom: 8px; }
</style>
</head>
<body>

<div class="top-bar">
  <h1>HDM Validation Dashboard</h1>
  <div class="controls">
    <label>Split:</label>
    <select id="splitSelect" onchange="changeSplit()"></select>
  </div>
</div>

<div class="tab-bar">
  <div class="tab active" data-tab="predictions" onclick="switchTab('predictions')">Test Predictions</div>
  <div class="tab" data-tab="shots" onclick="switchTab('shots')">Few-Shot Examples (P &amp; N)</div>
</div>

<div class="split-stats" id="splitStats"></div>

<div class="main">
  <div class="sidebar" id="sidebar"></div>
  <div class="detail-panel" id="detailPanel">
    <div class="detail-empty">Select an item from the sidebar to inspect</div>
  </div>
</div>

<script>
let allSplits = [];
let allShots = [];
let currentSplit = 0;
let currentTab = "predictions";
let currentFilter = "ALL";
let filteredItems = [];
let selectedIdx = -1;
let audioCache = {};

async function init() {
  const [splitsResp, shotsResp] = await Promise.all([
    fetch("/api/splits"),
    fetch("/api/shots")
  ]);
  allSplits = await splitsResp.json();
  allShots = await shotsResp.json();

  const sel = document.getElementById("splitSelect");
  allSplits.forEach((s, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.text = `Split ${i+1} (F1=${s.f1.toFixed(4)})`;
    sel.appendChild(opt);
  });

  changeSplit();
}

function switchTab(tab) {
  currentTab = tab;
  currentFilter = "ALL";
  document.querySelectorAll(".tab").forEach(t => t.classList.remove("active"));
  document.querySelector(`.tab[data-tab="${tab}"]`).classList.add("active");
  selectedIdx = -1;
  updateStats();
  applyFilter();
  document.getElementById("detailPanel").innerHTML = '<div class="detail-empty">Select an item from the sidebar to inspect</div>';
}

function changeSplit() {
  currentSplit = parseInt(document.getElementById("splitSelect").value);
  selectedIdx = -1;
  updateStats();
  applyFilter();
  document.getElementById("detailPanel").innerHTML = '<div class="detail-empty">Select an item from the sidebar to inspect</div>';
}

function updateStats() {
  const statsEl = document.getElementById("splitStats");
  if (currentTab === "predictions") {
    const s = allSplits[currentSplit];
    statsEl.innerHTML = `
      <div class="stat">Total: <b>${s.n_test}</b></div>
      <div class="stat">F1: <b>${s.f1.toFixed(4)}</b></div>
      <div class="stat" style="color:#4caf50">TP: <b>${s.n_tp}</b></div>
      <div class="stat" style="color:#f44336">FP: <b>${s.n_fp}</b></div>
      <div class="stat" style="color:#ff9800">FN: <b>${s.n_fn}</b></div>
      <div class="stat" style="color:#64b5f6">TN: <b>${s.n_tn}</b></div>
      <div style="flex:1"></div>
      <div class="filter-chips">
        <div class="chip chip-all ${currentFilter==='ALL'?'active':''}" data-filter="ALL" onclick="setFilter('ALL')">All</div>
        <div class="chip chip-tp ${currentFilter==='TP'?'active':''}" data-filter="TP" onclick="setFilter('TP')">TP</div>
        <div class="chip chip-fp ${currentFilter==='FP'?'active':''}" data-filter="FP" onclick="setFilter('FP')">FP</div>
        <div class="chip chip-fn ${currentFilter==='FN'?'active':''}" data-filter="FN" onclick="setFilter('FN')">FN</div>
        <div class="chip chip-tn ${currentFilter==='TN'?'active':''}" data-filter="TN" onclick="setFilter('TN')">TN</div>
      </div>
    `;
  } else {
    const sh = allShots[currentSplit];
    statsEl.innerHTML = `
      <div class="stat">Total shots: <b>${sh.shots.length}</b></div>
      <div class="stat" style="color:#4caf50">Positive (P): <b>${sh.n_pos}</b></div>
      <div class="stat" style="color:#f44336">Negative (N): <b>${sh.n_neg}</b></div>
      <div class="stat" style="color:#ff9800">Hard negatives: <b>${sh.n_hard_neg}</b></div>
      <div class="stat" style="color:#64b5f6">Random negatives: <b>${sh.n_random_neg}</b></div>
      <div style="flex:1"></div>
      <div class="filter-chips">
        <div class="chip chip-all ${currentFilter==='ALL'?'active':''}" data-filter="ALL" onclick="setFilter('ALL')">All</div>
        <div class="chip chip-pos ${currentFilter==='POS'?'active':''}" data-filter="POS" onclick="setFilter('POS')">Positive</div>
        <div class="chip chip-neg ${currentFilter==='NEG'?'active':''}" data-filter="NEG" onclick="setFilter('NEG')">Negative</div>
      </div>
    `;
  }
}

function setFilter(f) {
  currentFilter = f;
  applyFilter();
  updateStats();
}

function applyFilter() {
  if (currentTab === "predictions") {
    const items = allSplits[currentSplit].items;
    filteredItems = currentFilter === "ALL" ? items : items.filter(it => it.category === currentFilter);
  } else {
    const shots = allShots[currentSplit].shots;
    if (currentFilter === "ALL") filteredItems = shots;
    else if (currentFilter === "POS") filteredItems = shots.filter(s => s.label === 1);
    else if (currentFilter === "NEG") filteredItems = shots.filter(s => s.label === 0);
    else filteredItems = shots;
  }
  selectedIdx = -1;
  renderSidebar();
}

function renderSidebar() {
  const sb = document.getElementById("sidebar");
  if (filteredItems.length === 0) {
    sb.innerHTML = '<div style="padding:20px;color:#555;text-align:center;">No items in this category</div>';
    return;
  }

  if (currentTab === "predictions") {
    sb.innerHTML = filteredItems.map((it, i) => `
      <div class="sidebar-item ${i === selectedIdx ? 'active' : ''}" onclick="selectItem(${i})">
        <span class="sidebar-badge badge-${it.category.toLowerCase()}">${it.category}</span>
        <div class="sidebar-info">
          <div class="sidebar-meeting">${it.meeting_id} &mdash; ${it.speaker}</div>
          <div class="sidebar-detail">${it.text || '(no transcript)'}</div>
          <div class="sidebar-conf">Conf: ${(it.probability * 100).toFixed(1)}% &middot; ${formatTime(it.sample_time)}</div>
        </div>
      </div>
    `).join("");
  } else {
    sb.innerHTML = filteredItems.map((it, i) => `
      <div class="sidebar-item ${i === selectedIdx ? 'active' : ''}" onclick="selectItem(${i})">
        <span class="sidebar-badge badge-${it.label === 1 ? 'pos' : 'neg'}">${it.label === 1 ? 'P' : 'N'}</span>
        <div class="sidebar-info">
          <div class="sidebar-meeting">${it.meeting_id} &mdash; ${it.speaker}</div>
          <div class="sidebar-detail">${it.text || '(no transcript)'}</div>
          <div class="sidebar-conf">${it.source} &middot; ${formatTime(it.sample_time)}</div>
        </div>
      </div>
    `).join("");
  }
}

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = (sec % 60).toFixed(1);
  return m + ":" + s.padStart(4, '0');
}

async function selectItem(i) {
  selectedIdx = i;
  renderSidebar();

  const it = filteredItems[i];
  const panel = document.getElementById("detailPanel");
  panel.innerHTML = '<div class="detail-empty">Loading audio...</div>';

  const cacheKey = it.meeting_id + "_" + it.sample_time;
  if (!audioCache[cacheKey]) {
    const resp = await fetch("/api/audio?meeting=" + it.meeting_id + "&time=" + it.sample_time);
    audioCache[cacheKey] = await resp.json();
  }
  const audio = audioCache[cacheKey];

  if (currentTab === "predictions") {
    renderPredictionDetail(it, audio, i);
  } else {
    renderShotDetail(it, audio, i);
  }
}

function renderPredictionDetail(it, audio, i) {
  const panel = document.getElementById("detailPanel");
  const catColors = { TP: "#1b5e20", FP: "#b71c1c", FN: "#e65100", TN: "#1a3a5c" };
  const confPct = (it.probability * 100).toFixed(1);
  const confClass = it.probability > 0.7 ? "high" : it.probability > 0.3 ? "mid" : "low";

  panel.innerHTML = `
    <div class="detail-card">
      <div class="detail-header">
        <div class="detail-title">#${it.idx + 1} &mdash; ${it.meeting_id}</div>
        <div class="detail-category" style="background:${catColors[it.category]}">${it.category}</div>
      </div>

      <div class="meta-grid">
        <div class="meta-box"><div class="meta-label">Meeting</div><div class="meta-value">${it.meeting_id}</div></div>
        <div class="meta-box"><div class="meta-label">Speaker</div><div class="meta-value">${it.speaker}</div></div>
        <div class="meta-box"><div class="meta-label">Time</div><div class="meta-value">${formatTime(it.sample_time)}</div></div>
      </div>

      <div class="label-row">
        <div class="label-box">
          <div class="label-title">Ground Truth</div>
          <div class="label-val ${it.true_label === 1 ? 'label-p' : 'label-n'}">${it.true_label === 1 ? 'P (HDM)' : 'N (Not HDM)'}</div>
        </div>
        <div class="label-box">
          <div class="label-title">Model Prediction</div>
          <div class="label-val ${it.prediction === 1 ? 'label-p' : 'label-n'}">${it.prediction === 1 ? 'P (HDM)' : 'N (Not HDM)'}</div>
        </div>
      </div>

      <div class="confidence-bar">
        <div class="conf-label">Model Confidence (P(HDM) = ${confPct}%)</div>
        <div class="conf-track">
          <div class="conf-fill ${confClass}" style="width:${Math.max(5, confPct)}%">${confPct}%</div>
        </div>
      </div>

      <div class="transcript-box">
        ${it.text ? '&ldquo;' + escHtml(it.text) + '&rdquo;' : '<span class="no-transcript">No transcript available</span>'}
      </div>

      <div class="audio-section">
        <div class="audio-title">Full Extended Clip (~12s)</div>
        <div class="audio-desc">4s context + 4s HDM segment + 4s after</div>
        <audio controls src="data:audio/wav;base64,${audio.extended}"></audio>
      </div>
      <div class="audio-section">
        <div class="audio-title">HDM Segment Only (4s)</div>
        <div class="audio-desc">The moment being classified</div>
        <audio controls src="data:audio/wav;base64,${audio.hdm}"></audio>
      </div>
      <div class="audio-section">
        <div class="audio-title">Context Before (4s)</div>
        <div class="audio-desc">What was happening just before</div>
        <audio controls src="data:audio/wav;base64,${audio.context}"></audio>
      </div>

      <div class="nav-buttons">
        <button class="nav-btn" onclick="navPrev()" ${i === 0 ? 'disabled' : ''}>&larr; Previous</button>
        <button class="nav-btn" onclick="navNext()" ${i === filteredItems.length - 1 ? 'disabled' : ''}>Next &rarr;</button>
      </div>
      <div class="keyboard-hint">Keyboard: &larr;/&rarr; navigate, 1-5 switch splits</div>
    </div>
  `;
}

function renderShotDetail(it, audio, i) {
  const panel = document.getElementById("detailPanel");
  const isPos = it.label === 1;
  const badgeColor = isPos ? "#1b5e20" : "#b71c1c";
  const labelText = isPos ? "P (Positive — HDM)" : "N (Negative — Not HDM)";

  let sourceClass = "source-random";
  if (it.source.includes("verified")) sourceClass = "source-verified";
  else if (it.source.includes("hard")) sourceClass = "source-hard";
  else if (it.source.includes("unverified")) sourceClass = "source-unverified";

  // Find the order position in the full shots list
  const allShots2 = allShots[currentSplit].shots;
  const orderIdx = allShots2.findIndex(s => s.idx === it.idx);

  panel.innerHTML = `
    <div class="detail-card">
      <div class="detail-header">
        <div class="detail-title">Shot Example &mdash; ${it.meeting_id}</div>
        <div class="detail-category" style="background:${badgeColor}">${isPos ? 'P' : 'N'}</div>
      </div>

      <div class="shot-order">Prompt order: #${orderIdx + 1} of ${allShots2.length} &middot; Dataset index: ${it.idx}</div>
      <span class="source-tag ${sourceClass}">${it.source}</span>

      <div class="meta-grid">
        <div class="meta-box"><div class="meta-label">Meeting</div><div class="meta-value">${it.meeting_id}</div></div>
        <div class="meta-box"><div class="meta-label">Speaker</div><div class="meta-value">${it.speaker}</div></div>
        <div class="meta-box"><div class="meta-label">Time</div><div class="meta-value">${formatTime(it.sample_time)}</div></div>
      </div>

      <div class="label-row">
        <div class="label-box">
          <div class="label-title">Label Given to Model</div>
          <div class="label-val ${isPos ? 'label-p' : 'label-n'}">${labelText}</div>
        </div>
      </div>

      <div class="transcript-box">
        ${it.text ? '&ldquo;' + escHtml(it.text) + '&rdquo;' : '<span class="no-transcript">No transcript (negative example — full mix audio)</span>'}
      </div>

      <div class="audio-section">
        <div class="audio-title">Full Extended Clip (~12s)</div>
        <div class="audio-desc">This is the audio sent to the model as a few-shot example</div>
        <audio controls src="data:audio/wav;base64,${audio.extended}"></audio>
      </div>
      <div class="audio-section">
        <div class="audio-title">Core Segment (4s)</div>
        <div class="audio-desc">${isPos ? 'The hearing difficulty moment' : 'The non-HDM moment'}</div>
        <audio controls src="data:audio/wav;base64,${audio.hdm}"></audio>
      </div>
      <div class="audio-section">
        <div class="audio-title">Context Before (4s)</div>
        <div class="audio-desc">Preceding audio for context</div>
        <audio controls src="data:audio/wav;base64,${audio.context}"></audio>
      </div>

      <div class="nav-buttons">
        <button class="nav-btn" onclick="navPrev()" ${i === 0 ? 'disabled' : ''}>&larr; Previous</button>
        <button class="nav-btn" onclick="navNext()" ${i === filteredItems.length - 1 ? 'disabled' : ''}>Next &rarr;</button>
      </div>
      <div class="keyboard-hint">Keyboard: &larr;/&rarr; navigate, 1-5 switch splits</div>
    </div>
  `;
}

function escHtml(s) {
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function navPrev() { if (selectedIdx > 0) selectItem(selectedIdx - 1); }
function navNext() { if (selectedIdx < filteredItems.length - 1) selectItem(selectedIdx + 1); }

document.addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "ArrowLeft") navPrev();
  else if (e.key === "ArrowRight") navNext();
  else if (e.key >= "1" && e.key <= "5") {
    document.getElementById("splitSelect").value = parseInt(e.key) - 1;
    changeSplit();
  }
});

init();
</script>
</body>
</html>"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/" or path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

        elif path == "/api/splits":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(split_data).encode())

        elif path == "/api/shots":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(shot_data).encode())

        elif path == "/api/audio":
            meeting = params.get("meeting", [None])[0]
            time = float(params.get("time", [0])[0])
            if meeting:
                audio = get_audio_b64(meeting, time)
                if audio:
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    self.wfile.write(json.dumps(audio).encode())
                    return
            self.send_response(404)
            self.end_headers()

        else:
            self.send_response(404)
            self.end_headers()


def main():
    print("HDM Validation Dashboard")
    print("=" * 50)
    print(f"Loading results from {RESULTS_FILE}...")
    load_data()

    print(f"\nStarting server on http://localhost:{PORT}")
    print("Open in your browser to explore predictions with audio playback.")
    print("Press Ctrl+C to stop.\n")

    server = HTTPServer(("127.0.0.1", PORT), Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


if __name__ == "__main__":
    main()
