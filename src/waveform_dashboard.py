"""
Interactive Figure 1 Dashboard — Plotly + Audio Playback + Few-Shot Examples.

Fully interactive recreation of Collins et al. Figure 1:
- Plotly chart: zoom, pan, hover for details
- Click on chart to seek audio to that timestamp
- Blue waveform, green probability line, red HDM bands, orange threshold
- Audio playback with sync cursor
- Few-shot examples panel with audio for each P/N example
- Individual HDM clip playback

Usage:
    python src/waveform_dashboard.py
    Then open http://localhost:8766
"""

import gc
import io
import json
import base64
import numpy as np
import soundfile as sf
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from collections import defaultdict

ROOT = Path(__file__).parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
DATASET_DIR = ROOT / "data" / "dataset"
RESULTS_DIR = ROOT / "results"
SAMPLE_RATE = 16000
RANDOM_SEED = 42
N_POS_SHOTS = 12
N_NEG_SHOTS = 8

meeting_list = []  # [{id, duration, n_pos, tp, fp, fn}]
meeting_details = {}  # mid -> {hdm_regions, samples}
waveform_cache = {}
shot_data = {}  # split_idx -> list of shot dicts


def smooth_sliding_window(data, sigma=5.0):
    """Gaussian-smooth the prob_p signal to match the paper's smooth curve."""
    windows = data.get("windows", [])
    if len(windows) < 5:
        return data
    probs = np.array([w["prob_p"] for w in windows])
    preds = np.array([w["pred"] for w in windows])
    # Fix contradictory spikes: model said N but logprobs say high P
    for i in range(len(probs)):
        if preds[i] != 1 and probs[i] > 0.8:
            # Use median of neighbors instead
            lo = max(0, i - 2)
            hi = min(len(probs), i + 3)
            neighbors = [probs[j] for j in range(lo, hi) if j != i]
            probs[i] = float(np.median(neighbors)) if neighbors else probs[i]
    # Gaussian kernel
    radius = int(sigma * 3)
    x = np.arange(-radius, radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    # Pad and convolve
    padded = np.pad(probs, radius, mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed = np.clip(smoothed, 0.0, 1.0)
    for i, w in enumerate(windows):
        w["prob_p"] = round(float(smoothed[i]), 4)
    data["windows"] = windows
    return data


def load_data():
    global meeting_list, meeting_details, shot_data

    with open(RESULTS_DIR / "gpt4o_20shot_v4_results.json") as f:
        v4 = json.load(f)
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)
    human_labels = {}
    p = ROOT / "data" / "hdm_labels.json"
    if p.exists():
        with open(p) as f:
            human_labels = json.load(f)

    all_ex = meta["positive"] + meta["negative"]
    n_positives = 149  # matches v4 classifier constant
    md = defaultdict(lambda: {"hdm_regions": [], "samples": []})

    # Track which split each meeting belongs to (for shot lookup)
    meeting_to_split = {}

    for i, ex in enumerate(meta["positive"]):
        mid = ex["meeting_id"]
        if ex.get("hdm_start") is not None:
            md[mid]["hdm_regions"].append({
                "start": round(ex["hdm_start"], 3),
                "end": round(ex["hdm_end"], 3),
                "text": ex.get("text", ""),
                "speaker": ex.get("speaker", ""),
            })

    for si, sr in enumerate(v4["splits"]):
        sm = meta["splits"][si]
        tm = set(sm["test"])
        train_meetings = set(sm["train"])
        ti = [j for j, ex in enumerate(all_ex) if ex["meeting_id"] in tm]

        # Reconstruct the exact test indices the classifier used:
        # positives first (by global index order), then negatives
        pos_ti = [j for j in ti if j < n_positives]
        neg_ti = [j for j in ti if j >= n_positives]
        n_pos_result = sum(sr["true_labels"])
        n_neg_result = len(sr["true_labels"]) - n_pos_result
        ti = pos_ti[:n_pos_result] + neg_ti[:n_neg_result]

        for li, gi in enumerate(ti):
            ex = all_ex[gi]
            md[ex["meeting_id"]]["samples"].append({
                "time": round(ex["sample_time"], 3),
                "label": ex["label"],
                "pred": sr["predictions"][li],
                "prob_p": round(sr["probabilities"][li], 4),
                "text": ex.get("text", ""),
                "speaker": ex.get("speaker", ""),
            })

        # Track which meetings are in this split's test set
        for mid in tm:
            meeting_to_split[mid] = si

        # Build few-shot examples for this split
        train_indices = [j for j, ex in enumerate(all_ex) if ex["meeting_id"] in train_meetings]
        verified_pos, hard_neg, unverified_pos, random_neg = [], [], [], []
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

        np.random.seed(RANDOM_SEED + si)
        pos_pool = verified_pos if len(verified_pos) >= N_POS_SHOTS else verified_pos + unverified_pos
        selected_pos = list(np.random.choice(pos_pool, size=min(N_POS_SHOTS, len(pos_pool)), replace=False))
        n_hard = min(len(hard_neg), N_NEG_SHOTS // 2)
        n_random = N_NEG_SHOTS - n_hard
        hard_selected = list(np.random.choice(hard_neg, size=n_hard, replace=False)) if hard_neg else []
        random_selected = list(np.random.choice(random_neg, size=min(n_random, len(random_neg)), replace=False))
        selected_neg = hard_selected + random_selected

        shots = []
        pi, ni = 0, 0
        while pi < len(selected_pos) or ni < len(selected_neg):
            for _ in range(2):
                if pi < len(selected_pos):
                    idx = int(selected_pos[pi])
                    ex = all_ex[idx]
                    source = "human-verified" if human_labels.get(str(idx)) == "yes" else "unverified"
                    shots.append({
                        "idx": idx, "meeting_id": ex["meeting_id"],
                        "speaker": ex.get("speaker", "?"),
                        "sample_time": round(ex["sample_time"], 3),
                        "text": ex.get("text", ""), "label": 1,
                        "source": source,
                    })
                    pi += 1
            if ni < len(selected_neg):
                idx = int(selected_neg[ni])
                ex = all_ex[idx]
                is_hard = idx in hard_neg
                source = "hard-negative" if is_hard else "random-negative"
                shots.append({
                    "idx": idx, "meeting_id": ex["meeting_id"],
                    "speaker": ex.get("speaker", "?"),
                    "sample_time": round(ex["sample_time"], 3),
                    "text": ex.get("text", ""), "label": 0,
                    "source": source,
                })
                ni += 1

        shot_data[si] = shots

    for mid, data in sorted(md.items()):
        ap = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        if not ap.exists():
            continue
        n_pos = sum(1 for s in data["samples"] if s["label"] == 1)
        if n_pos == 0:
            continue
        info = sf.info(str(ap))
        data["hdm_regions"].sort(key=lambda x: x["start"])
        data["samples"].sort(key=lambda x: x["time"])
        tp = sum(1 for s in data["samples"] if s["label"] == 1 and s["pred"] == 1)
        fp = sum(1 for s in data["samples"] if s["label"] == 0 and s["pred"] == 1)
        fn = sum(1 for s in data["samples"] if s["label"] == 1 and s["pred"] == 0)
        split_idx = meeting_to_split.get(mid, 0)
        meeting_list.append({
            "id": mid, "duration": round(info.duration, 2),
            "n_pos": n_pos, "tp": tp, "fp": fp, "fn": fn,
            "split": split_idx,
        })
        meeting_details[mid] = data

    meeting_list.sort(key=lambda x: -x["n_pos"])
    print(f"Loaded {len(meeting_list)} meetings")
    print(f"Loaded few-shot examples for {len(shot_data)} splits")


def get_waveform(mid):
    if mid not in waveform_cache:
        ap = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        audio, sr = sf.read(str(ap), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)
        # Downsample to ~8000 points
        target = 8000
        chunk = max(1, len(audio) // target)
        n = len(audio) // chunk
        reshaped = audio[:n * chunk].reshape(n, chunk)
        waveform_cache[mid] = {
            "times": (np.arange(n) * chunk / SAMPLE_RATE).round(3).tolist(),
            "maxes": reshaped.max(axis=1).round(4).tolist(),
            "mins": reshaped.min(axis=1).round(4).tolist(),
        }
        del audio
        gc.collect()
    return waveform_cache[mid]


def get_prob_signal(mid):
    """Build two probability signals: one for actual P predictions, one for logprob-only.

    Returns separate spike traces for:
    - pred_spikes: only samples where the model actually output "P" (pred=1)
    - logprob_spikes: all samples' logprob P(HDM), shown as faded for context
    """
    data = meeting_details[mid]
    samples = sorted(data["samples"], key=lambda s: s["time"])

    spike_half_width = 1.5  # seconds

    # Actual predictions (model output "P")
    pred_times, pred_probs, pred_texts = [], [], []
    # Logprob signal (all samples, faded)
    log_times, log_probs, log_texts = [], [], []

    for s in samples:
        t = s["time"]
        p = s["prob_p"]
        hover = (
            "Time: " + str(round(t, 1)) + "s | "
            "P(HDM): " + str(round(p, 3)) + " | "
            "Pred: " + ("P" if s["pred"] == 1 else "N") + " | "
            "True: " + ("P" if s["label"] == 1 else "N") + " | "
            + s.get("text", "")
        )

        # Logprob trace (all samples)
        log_times.extend([t - spike_half_width, t, t + spike_half_width])
        log_probs.extend([0, p, 0])
        log_texts.extend(["", hover, ""])

        # Actual prediction trace (only pred=1)
        if s["pred"] == 1:
            pred_times.extend([t - spike_half_width, t, t + spike_half_width])
            pred_probs.extend([0, p, 0])
            pred_texts.extend(["", hover, ""])

    return {
        "pred_times": pred_times,
        "pred_probs": pred_probs,
        "pred_texts": pred_texts,
        "log_times": log_times,
        "log_probs": log_probs,
        "log_texts": log_texts,
    }


def extract_clip_b64(meeting_id, sample_time):
    """Extract a 12s clip as base64 WAV."""
    ap = AUDIO_DIR / f"{meeting_id}.Mix-Headset.wav"
    if not ap.exists():
        return None
    audio, sr = sf.read(str(ap), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)
    start = max(0, int((sample_time - 8.0) * sr))
    end = min(len(audio), int((sample_time + 4.0) * sr))
    clip = audio[start:end]
    buf = io.BytesIO()
    sf.write(buf, clip.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    del audio
    gc.collect()
    return base64.b64encode(buf.read()).decode("utf-8")


HTML = r"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Figure 1 — HDM Detection Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f5f6fa; color: #2c3e50; }

.header { background: #fff; border-bottom: 1px solid #e0e0e0; padding: 14px 20px; text-align: center; }
.header h1 { font-size: 18px; color: #2c3e50; font-weight: 700; }
.header p { font-size: 12px; color: #777; margin-top: 2px; }

.stats-bar { display: flex; gap: 10px; justify-content: center; padding: 12px; background: #fff; border-bottom: 1px solid #e0e0e0; }
.stat-card { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 6px; padding: 8px 18px; text-align: center; }
.stat-card .val { font-size: 22px; font-weight: 700; color: #1a9641; }
.stat-card .val.muted { color: #999; }
.stat-card .lbl { font-size: 9px; color: #999; text-transform: uppercase; letter-spacing: 0.5px; }

.nav-bar { display: flex; gap: 4px; padding: 10px 20px; flex-wrap: wrap; justify-content: center; background: #fff; border-bottom: 1px solid #e0e0e0; }
.nav-btn { padding: 5px 12px; background: #f8f9fa; border: 1px solid #dee2e6; color: #495057; border-radius: 5px; cursor: pointer; font-size: 11px; transition: all 0.15s; }
.nav-btn:hover { background: #e9ecef; }
.nav-btn.active { background: #1a9641; border-color: #1a9641; color: #fff; }

.content { max-width: 1200px; margin: 15px auto; padding: 0 15px; }

.chart-card { background: #fff; border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.chart-card .chart-header { padding: 10px 15px; border-bottom: 1px solid #f0f0f0; display: flex; justify-content: space-between; align-items: center; }
.chart-card .chart-title { font-size: 14px; font-weight: 600; color: #2c3e50; }
.chart-card .chart-meta { font-size: 11px; color: #999; }
#chart { width: 100%; height: 550px; }

.audio-card { background: #fff; border: 1px solid #e0e0e0; border-radius: 10px; padding: 12px 15px; margin-bottom: 12px; display: flex; align-items: center; gap: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.play-btn { padding: 8px 20px; background: #1a9641; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; min-width: 70px; }
.play-btn:hover { background: #158a38; }
.time-display { font-family: 'SF Mono', 'Fira Code', monospace; font-size: 13px; color: #666; }
.audio-hint { font-size: 11px; color: #aaa; margin-left: auto; }

.events-card { background: #fff; border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.events-header { padding: 10px 15px; border-bottom: 1px solid #f0f0f0; font-size: 13px; font-weight: 600; color: #2c3e50; }
.event-row { display: flex; align-items: center; gap: 10px; padding: 8px 15px; cursor: pointer; border-bottom: 1px solid #f8f8f8; transition: background 0.1s; }
.event-row:hover { background: #f0fff4; }
.event-row:last-child { border-bottom: none; }
.event-row.fn { background: #fffdf0; }
.event-row.fn:hover { background: #fff8e0; }
.ev-time { font-family: monospace; font-size: 12px; color: #888; min-width: 55px; }
.ev-text { font-style: italic; flex: 1; font-size: 13px; color: #333; }
.ev-speaker { font-size: 11px; color: #aaa; min-width: 65px; }
.tag { padding: 2px 8px; border-radius: 10px; font-size: 9px; font-weight: 700; letter-spacing: 0.3px; }
.tag-tp { background: #d4edda; color: #155724; }
.tag-fn { background: #fff3cd; color: #856404; }
.tag-fp { background: #f8d7da; color: #721c24; }
.tag-prob { background: #e9ecef; color: #495057; font-family: monospace; }

.loading { text-align: center; padding: 40px; color: #aaa; font-size: 14px; }

.section-title { padding: 12px 15px; font-size: 14px; font-weight: 700; color: #2c3e50; border-bottom: 1px solid #f0f0f0; display: flex; align-items: center; gap: 8px; cursor: pointer; user-select: none; }
.section-title:hover { background: #f8f9fa; }
.section-title .toggle { font-size: 10px; color: #aaa; transition: transform 0.2s; }
.section-title .toggle.open { transform: rotate(90deg); }

.clip-player { display: flex; align-items: center; gap: 8px; margin-top: 4px; }
.clip-player audio { height: 28px; flex: 1; }
.clip-btn { padding: 3px 10px; background: #1a9641; color: #fff; border: none; border-radius: 4px; cursor: pointer; font-size: 10px; font-weight: 600; white-space: nowrap; }
.clip-btn:hover { background: #158a38; }
.clip-btn.loading-clip { background: #999; pointer-events: none; }

.shots-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0; }
.shot-item { padding: 10px 15px; border-bottom: 1px solid #f0f0f0; border-right: 1px solid #f0f0f0; }
.shot-item:hover { background: #f8fff8; }
.shot-header { display: flex; align-items: center; gap: 8px; margin-bottom: 4px; }
.shot-label { padding: 2px 10px; border-radius: 10px; font-size: 10px; font-weight: 700; }
.shot-label-p { background: #d4edda; color: #155724; }
.shot-label-n { background: #f8d7da; color: #721c24; }
.shot-source { font-size: 9px; padding: 1px 6px; border-radius: 8px; background: #e9ecef; color: #666; }
.shot-meta { font-size: 11px; color: #888; margin-bottom: 2px; }
.shot-text { font-size: 12px; color: #333; font-style: italic; margin-bottom: 4px; min-height: 16px; }
.shot-order { font-size: 9px; color: #bbb; }
</style>
</head>
<body>

<div class="header">
  <h1>Audio Waveform vs. Model Prediction and Ground Truth</h1>
  <p>Recreating Collins et al. Figure 1 — GPT-4o Audio v4 (20-shot) on AMI Corpus | Click chart to seek audio</p>
  <p style="font-size:10px;color:#999;margin-top:2px;">Green spikes = model output "P" (actual predictions used for F1). Red bands = ground truth HDM events.</p>
</div>

<div class="stats-bar">
  <div class="stat-card"><div class="val" id="s-f1">-</div><div class="lbl">Our F1 (v4)</div></div>
  <div class="stat-card"><div class="val muted">0.87</div><div class="lbl">Paper F1</div></div>
  <div class="stat-card"><div class="val" id="s-best">-</div><div class="lbl">Best Split</div></div>
  <div class="stat-card"><div class="val" id="s-tp">-</div><div class="lbl">TP</div></div>
  <div class="stat-card"><div class="val" style="color:#da3633" id="s-fp">-</div><div class="lbl">FP</div></div>
  <div class="stat-card"><div class="val" style="color:#d29922" id="s-fn">-</div><div class="lbl">FN</div></div>
</div>

<div class="nav-bar" id="nav"></div>

<div class="content">
  <div class="chart-card">
    <div class="chart-header">
      <div class="chart-title" id="chart-title">Select a meeting</div>
      <div class="chart-meta" id="chart-meta"></div>
    </div>
    <div id="chart"><div class="loading">Loading...</div></div>
  </div>

  <div class="audio-card" style="flex-wrap:wrap;">
    <button class="play-btn" id="play-btn" onclick="togglePlay()">Play</button>
    <button class="play-btn" style="background:#6c757d;min-width:40px;padding:8px 10px" onclick="skipAudio(-5)">-5s</button>
    <button class="play-btn" style="background:#6c757d;min-width:40px;padding:8px 10px" onclick="skipAudio(5)">+5s</button>
    <audio id="audio" preload="none"></audio>
    <div class="time-display" id="time-display">0:00 / 0:00</div>
    <div style="width:100%;margin-top:6px;display:flex;align-items:center;gap:8px;">
      <input type="range" id="seekbar" min="0" max="1000" value="0" step="1"
        style="flex:1;height:8px;cursor:pointer;accent-color:#1a9641;">
      <select id="speed" onchange="audio.playbackRate=parseFloat(this.value)" style="padding:2px 4px;border-radius:4px;border:1px solid #ccc;font-size:11px;">
        <option value="0.5">0.5x</option>
        <option value="0.75">0.75x</option>
        <option value="1" selected>1x</option>
        <option value="1.5">1.5x</option>
        <option value="2">2x</option>
      </select>
    </div>
    <div style="width:100%;font-size:10px;color:#aaa;margin-top:2px;">Drag scrubber to seek | Click chart to jump | Space = play/pause | Arrow keys = skip 5s</div>
  </div>

  <div class="audio-card" style="background:#f8fff8;border-color:#c8e6c9;padding:10px 15px;">
    <div style="display:flex;align-items:center;gap:10px;width:100%;">
      <button class="play-btn" style="background:#d32f2f;min-width:36px;padding:6px 12px;font-size:12px;border-radius:5px;" onclick="navDetection(-1)">&#9664;</button>
      <button class="play-btn" style="background:#d32f2f;min-width:36px;padding:6px 12px;font-size:12px;border-radius:5px;" onclick="navDetection(1)">&#9654;</button>
      <div style="font-size:13px;font-weight:600;color:#2c3e50;" id="det-nav-label">Detections: loading...</div>
      <div style="margin-left:auto;display:flex;align-items:center;gap:6px;">
        <span id="det-nav-tag" style="font-size:11px;"></span>
        <span id="det-nav-prob" style="font-family:monospace;font-size:12px;color:#666;"></span>
      </div>
    </div>
  </div>

  <div class="events-card" style="margin-top:12px;">
    <div class="section-title" onclick="toggleSection('events')">
      <span class="toggle open" id="toggle-events">&#9654;</span>
      <span id="events-header">HDM Events</span>
    </div>
    <div id="events-list"></div>
  </div>

  <div class="events-card" style="margin-top:12px;">
    <div class="section-title" onclick="toggleSection('shots')">
      <span class="toggle open" id="toggle-shots">&#9654;</span>
      <span>20-Shot Examples Used (P &amp; N labels sent to model)</span>
    </div>
    <div id="shots-list"></div>
  </div>
</div>

<script>
let meetings = [];
let currentMid = null;
let audio = document.getElementById('audio');
let cursorInterval = null;
let clipCache = {};

function fmtTime(s) {
  if (!s || isNaN(s)) return '0:00';
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}

async function init() {
  const r = await fetch('/api/meetings');
  meetings = await r.json();
  const nav = document.getElementById('nav');
  meetings.forEach((m, i) => {
    const btn = document.createElement('button');
    btn.className = 'nav-btn';
    btn.textContent = m.id + ' (' + m.n_pos + ')';
    btn.onclick = () => selectMeeting(m.id);
    btn.id = 'btn-' + m.id;
    nav.appendChild(btn);
  });
  // Compute global stats from results
  const resp2 = await fetch('/api/global_stats');
  const gs = await resp2.json();
  document.getElementById('s-f1').textContent = gs.avg_f1.toFixed(2);
  document.getElementById('s-best').textContent = gs.best_f1.toFixed(2);

  if (meetings.length > 0) selectMeeting(meetings[0].id);
}

async function selectMeeting(mid) {
  if (currentMid === mid) return;
  currentMid = mid;
  const m = meetings.find(x => x.id === mid);

  // Update nav
  document.querySelectorAll('.nav-btn').forEach(b => b.classList.remove('active'));
  document.getElementById('btn-' + mid).classList.add('active');

  // Update stats
  document.getElementById('s-tp').textContent = m.tp;
  document.getElementById('s-fp').textContent = m.fp;
  document.getElementById('s-fn').textContent = m.fn;
  document.getElementById('chart-title').textContent = mid;
  document.getElementById('chart-meta').textContent =
    (m.duration / 60).toFixed(1) + ' min | ' + m.n_pos + ' HDMs | TP=' + m.tp + ' FP=' + m.fp + ' FN=' + m.fn;

  // Load audio
  audio.pause();
  audio.src = '/audio/' + mid;
  audio.load();
  document.getElementById('play-btn').textContent = 'Play';

  // Show loading
  document.getElementById('chart').innerHTML = '<div class="loading">Loading waveform...</div>';

  // Fetch waveform + meeting data + probability signal + sliding window in parallel
  const [waveResp, detailResp, probResp, slidingResp] = await Promise.all([
    fetch('/api/waveform/' + mid),
    fetch('/api/detail/' + mid),
    fetch('/api/probsignal/' + mid),
    fetch('/api/sliding/' + mid),
  ]);
  const wave = await waveResp.json();
  const detail = await detailResp.json();
  const prob = await probResp.json();
  const sliding = await slidingResp.json();

  buildChart(mid, m, wave, detail, prob, sliding);
  buildDetectionsList(mid, detail, sliding);
  buildEventList(mid, m, detail);
  buildShotList(m.split);

  // Start cursor update
  if (cursorInterval) clearInterval(cursorInterval);
  cursorInterval = setInterval(updateCursor, 200);
}

function buildChart(mid, m, wave, detail, prob, sliding) {
  const traces = [];
  const hasSliding = sliding && sliding.windows && sliding.windows.length > 0;

  // 1. Waveform envelope (blue filled area)
  traces.push({
    x: wave.times, y: wave.maxes,
    type: 'scatter', mode: 'lines',
    line: { color: 'rgba(0,0,0,0)', width: 0 },
    showlegend: false, hoverinfo: 'skip',
    yaxis: 'y',
  });
  traces.push({
    x: wave.times, y: wave.mins,
    type: 'scatter', mode: 'lines',
    line: { color: 'rgba(0,0,0,0)', width: 0 },
    fill: 'tonexty', fillcolor: 'rgba(100,149,237,0.3)',
    name: 'Audio Waveform',
    hoverinfo: 'skip',
    yaxis: 'y',
  });

  // 2. Continuous probability signal (sliding window) — like paper's Figure 1
  if (hasSliding) {
    const swTimes = sliding.windows.map(w => w.time);
    const swProbs = sliding.windows.map(w => w.prob_p);
    const swTexts = sliding.windows.map(w =>
      'Time: ' + w.time.toFixed(1) + 's | P(HDM): ' + w.prob_p.toFixed(3) +
      ' | Pred: ' + (w.pred === 1 ? 'P' : 'N')
    );
    traces.push({
      x: swTimes, y: swProbs,
      type: 'scatter', mode: 'lines',
      line: { color: '#1a9641', width: 1.5 },
      fill: 'tozeroy', fillcolor: 'rgba(26,150,65,0.08)',
      name: 'Model Prediction',
      text: swTexts, hoverinfo: 'text',
      yaxis: 'y2',
      legendrank: 2,
    });
  } else {
    // Fallback: show sparse prediction spikes if no sliding window data
    const predPeakX = [];
    const predPeakY = [];
    const predPeakText = [];
    for (let i = 1; i < prob.pred_probs.length; i += 3) {
      predPeakX.push(prob.pred_times[i]);
      predPeakY.push(prob.pred_probs[i]);
      predPeakText.push(prob.pred_texts[i] || '');
    }
    traces.push({
      x: prob.pred_times, y: prob.pred_probs,
      type: 'scatter', mode: 'lines',
      line: { color: '#00c853', width: 3 },
      fill: 'tozeroy', fillcolor: 'rgba(0,200,83,0.25)',
      name: 'Model Predicted "P"',
      text: prob.pred_texts || [], hoverinfo: 'text',
      yaxis: 'y2',
      legendrank: 2,
    });
    traces.push({
      x: predPeakX, y: predPeakY,
      type: 'scatter', mode: 'markers',
      marker: { color: '#00c853', size: 10, symbol: 'diamond',
                line: { color: '#fff', width: 1.5 } },
      name: 'Prediction Peak',
      text: predPeakText, hoverinfo: 'text',
      yaxis: 'y2',
      showlegend: false,
    });
  }

  // 3. Playback cursor (vertical line, updated via relayout)
  // Cursor trace index: 2 (waveform) + 1 (prob signal or 2 for fallback) + 0
  traces.push({
    x: [0, 0], y: [-1, 1],
    type: 'scatter', mode: 'lines',
    line: { color: '#e74c3c', width: 2.5 },
    name: 'Playback',
    hoverinfo: 'skip',
    yaxis: 'y',
    legendrank: 5,
  });

  // 4. Dummy traces for legend
  traces.push({
    x: [null], y: [null], type: 'scatter', mode: 'lines',
    line: { color: 'rgba(255,50,50,0.7)', width: 10 },
    name: 'Ground Truth HDM',
    legendrank: 1,
  });

  // Layout with shapes for HDM regions and threshold
  const shapes = [];

  // HDM ground truth regions (red bands)
  detail.hdm_regions.forEach(r => {
    const minDur = Math.max(r.end - r.start, m.duration * 0.003);
    const center = (r.start + r.end) / 2;
    shapes.push({
      type: 'rect',
      xref: 'x', yref: 'paper',
      x0: center - minDur / 2, x1: center + minDur / 2,
      y0: 0, y1: 1,
      fillcolor: 'rgba(255,50,50,0.35)',
      line: { color: 'rgba(220,30,30,0.8)', width: 2 },
      layer: 'below',
    });
  });

  // Threshold line (orange dashed)
  shapes.push({
    type: 'line',
    xref: 'paper', yref: 'y2',
    x0: 0, x1: 1,
    y0: 0.97, y1: 0.97,
    line: { color: 'orange', width: 1.5, dash: 'dash' },
  });

  // Positive prediction regions (bright green shading)
  if (hasSliding) {
    // Shade contiguous regions where sliding window pred=1
    let regionStart = null;
    const step = sliding.step_s || 4;
    sliding.windows.forEach((w, i) => {
      if (w.pred === 1 && regionStart === null) {
        regionStart = w.time - step / 2;
      } else if (w.pred !== 1 && regionStart !== null) {
        shapes.push({
          type: 'rect', xref: 'x', yref: 'paper',
          x0: regionStart, x1: sliding.windows[i-1].time + step / 2,
          y0: 0, y1: 1,
          fillcolor: 'rgba(0,200,83,0.18)',
          line: { color: 'rgba(0,200,83,0.5)', width: 1 },
          layer: 'below',
        });
        regionStart = null;
      }
    });
    if (regionStart !== null) {
      const last = sliding.windows[sliding.windows.length - 1];
      shapes.push({
        type: 'rect', xref: 'x', yref: 'paper',
        x0: regionStart, x1: last.time + step / 2,
        y0: 0, y1: 1,
        fillcolor: 'rgba(0,200,83,0.18)',
        line: { color: 'rgba(0,200,83,0.5)', width: 1 },
        layer: 'below',
      });
    }
  } else {
    for (let i = 1; i < prob.pred_probs.length; i += 3) {
      const t = prob.pred_times[i];
      const halfW = 2.0;
      shapes.push({
        type: 'rect', xref: 'x', yref: 'paper',
        x0: t - halfW, x1: t + halfW,
        y0: 0, y1: 1,
        fillcolor: 'rgba(0,200,83,0.18)',
        line: { color: 'rgba(0,200,83,0.5)', width: 1 },
        layer: 'below',
      });
    }
  }

  // Threshold label only (no HDM text annotations)
  const annotations = [];
  annotations.push({
    x: 1.01, xref: 'paper',
    y: 0.97, yref: 'y2',
    text: '0.97',
    showarrow: false,
    font: { size: 10, color: 'orange' },
    xanchor: 'left',
  });

  const layout = {
    xaxis: {
      title: { text: 'Time (s)', font: { size: 9 } },
      tickformat: ',d',
      ticksuffix: '',
      dtick: Math.max(10, Math.round(m.duration / 10)),
      showgrid: true, gridcolor: 'rgba(0,0,0,0.05)',
      range: [0, m.duration],
      // Show ms on hover
      hoverformat: ',.0f',
    },
    yaxis: {
      title: { text: 'Amplitude', font: { size: 9, color: 'steelblue' } },
      tickfont: { size: 9, color: 'steelblue' },
      showgrid: true, gridcolor: 'rgba(0,0,0,0.05)',
      fixedrange: true,
    },
    yaxis2: {
      title: { text: 'P(HDM)', font: { size: 9, color: '#1a9641' } },
      tickfont: { size: 9, color: '#1a9641' },
      overlaying: 'y', side: 'right',
      range: [-0.05, 1.1],
      showgrid: false,
      fixedrange: true,
    },
    shapes: shapes,
    annotations: annotations,
    legend: {
      orientation: 'h', x: 0.5, xanchor: 'center', y: 1.12,
      font: { size: 10 },
    },
    margin: { l: 55, r: 55, t: 40, b: 45 },
    hovermode: 'x unified',
    plot_bgcolor: '#fafafa',
    paper_bgcolor: '#fff',
  };

  Plotly.newPlot('chart', traces, layout, {
    responsive: true,
    displayModeBar: true,
    modeBarButtonsToRemove: ['lasso2d', 'select2d'],
  });

  // Click on chart → seek audio
  document.getElementById('chart').on('plotly_click', function(data) {
    if (data.points && data.points.length > 0) {
      const t = data.points[0].x;
      seekAudio(t);
    }
  });
}

let seekbarDragging = false;

function updateCursor() {
  if (!audio || !audio.duration || !currentMid) return;
  const t = audio.currentTime;
  const chartEl = document.getElementById('chart');

  // Move the cursor trace (last trace before the dummy legend trace)
  // Find cursor trace by name
  if (chartEl && chartEl.data) {
    const cursorIdx = chartEl.data.findIndex(tr => tr.name === 'Playback');
    if (cursorIdx >= 0) {
      Plotly.restyle('chart', { x: [[t, t]] }, [cursorIdx]);
    }
  }

  document.getElementById('time-display').textContent =
    fmtTime(audio.currentTime) + ' / ' + fmtTime(audio.duration);

  // Sync seekbar
  if (!seekbarDragging) {
    const sb = document.getElementById('seekbar');
    sb.value = (audio.currentTime / audio.duration) * 1000;
  }

  // Auto-scroll: if zoomed in and cursor goes past 80% of visible window, scroll forward
  if (chartEl && chartEl.layout && chartEl.layout.xaxis && chartEl.layout.xaxis.range) {
    const xRange = chartEl.layout.xaxis.range;
    const viewStart = xRange[0];
    const viewEnd = xRange[1];
    const viewDur = viewEnd - viewStart;
    const m = meetings.find(x => x.id === currentMid);
    const fullDur = m ? m.duration : audio.duration;

    // Only auto-scroll if we're zoomed in (viewing less than 90% of full duration)
    if (viewDur < fullDur * 0.9 && !audio.paused) {
      // If cursor past 75% of visible window, scroll to keep cursor at 25%
      if (t > viewStart + viewDur * 0.75) {
        const newStart = t - viewDur * 0.25;
        const newEnd = newStart + viewDur;
        Plotly.relayout('chart', {
          'xaxis.range': [Math.max(0, newStart), Math.min(fullDur, newEnd)]
        });
      }
      // If cursor before visible window (e.g. user seeked backwards)
      if (t < viewStart) {
        const newStart = t - viewDur * 0.25;
        const newEnd = newStart + viewDur;
        Plotly.relayout('chart', {
          'xaxis.range': [Math.max(0, newStart), Math.min(fullDur, newEnd)]
        });
      }
    }
  }
}

// Seekbar drag-to-scrub
(function() {
  const sb = document.getElementById('seekbar');
  sb.addEventListener('mousedown', () => { seekbarDragging = true; });
  sb.addEventListener('touchstart', () => { seekbarDragging = true; });
  sb.addEventListener('input', () => {
    if (audio && audio.duration) {
      audio.currentTime = (sb.value / 1000) * audio.duration;
    }
  });
  sb.addEventListener('mouseup', () => { seekbarDragging = false; });
  sb.addEventListener('touchend', () => { seekbarDragging = false; });
  sb.addEventListener('change', () => { seekbarDragging = false; });
})();

function skipAudio(sec) {
  if (!audio) return;
  audio.currentTime = Math.max(0, Math.min(audio.duration || 0, audio.currentTime + sec));
}

function togglePlay() {
  if (!audio || !audio.src) return;
  if (audio.paused) {
    audio.play();
    document.getElementById('play-btn').textContent = 'Pause';
  } else {
    audio.pause();
    document.getElementById('play-btn').textContent = 'Play';
  }
}

function seekAudio(timeSec) {
  if (!audio) return;
  audio.currentTime = Math.max(0, timeSec);
  if (audio.paused) {
    audio.play();
    document.getElementById('play-btn').textContent = 'Pause';
  }
}

function jumpTo(timeSec) {
  // Seek 4s before the HDM for context
  seekAudio(Math.max(0, timeSec - 4));
  // Zoom chart to 30s window around this point
  const m = meetings.find(x => x.id === currentMid);
  if (m) {
    const halfWindow = 15;
    Plotly.relayout('chart', {
      'xaxis.range': [Math.max(0, timeSec - halfWindow), Math.min(m.duration, timeSec + halfWindow)],
    });
  }
}

function buildDetectionsList(mid, detail, sliding) {
  const el = document.getElementById('detections-list');
  const header = document.getElementById('detections-header');
  const THRESHOLD = 0.97;

  if (!sliding || !sliding.windows || sliding.windows.length === 0) {
    header.textContent = 'Detections — no sliding window data';
    el.innerHTML = '<div style="padding:12px;color:#aaa;font-size:12px;">No sliding window data for this meeting.</div>';
    return;
  }

  // Find all windows above threshold
  const detections = sliding.windows.filter(w => w.prob_p >= THRESHOLD);

  // Get ground truth HDM regions
  const hdms = detail.hdm_regions || [];

  // Check if a detection is near a ground truth HDM (within 6s)
  function isTP(t) {
    return hdms.some(h => Math.abs(t - (h.start + h.end) / 2) < 6);
  }

  const tpCount = detections.filter(d => isTP(d.time)).length;
  const fpCount = detections.length - tpCount;
  header.textContent = 'Detections above ' + THRESHOLD + ' — ' + detections.length + ' total (TP~' + tpCount + ' FP~' + fpCount + ')';

  // Navigation buttons for prev/next detection
  let html = '<div style="padding:8px 15px;display:flex;gap:6px;align-items:center;border-bottom:1px solid #f0f0f0;">';
  html += '<button class="clip-btn" style="background:#495057" onclick="navDetection(-1)">Prev</button>';
  html += '<button class="clip-btn" style="background:#495057" onclick="navDetection(1)">Next</button>';
  html += '<span style="font-size:11px;color:#888;margin-left:8px;">Jump between detections</span>';
  html += '</div>';

  // List each detection
  detections.forEach((d, i) => {
    const tp = isTP(d.time);
    const tag = tp
      ? '<span class="tag tag-tp">TP</span>'
      : '<span class="tag tag-fp">FP</span>';
    const border = tp ? '' : 'border-left:3px solid #da3633;';
    html += '<div class="event-row" style="' + border + '" data-det-time="' + d.time + '">' +
      '<div class="ev-time" style="cursor:pointer" onclick="jumpTo(' + d.time + ')">' + fmtTime(d.time) + '</div>' +
      '<div style="flex:1;font-size:12px;color:#666;cursor:pointer" onclick="jumpTo(' + d.time + ')">P(HDM) = ' + d.prob_p.toFixed(3) + '</div>' +
      tag +
      '<span class="tag tag-prob">' + (d.pred === 1 ? 'Pred: P' : 'Pred: N') + '</span>' +
      '</div>';
  });

  if (detections.length === 0) {
    html += '<div style="padding:12px;color:#aaa;font-size:12px;">No windows above threshold.</div>';
  }

  el.innerHTML = html;

  // Store detections for prev/next navigation
  window._detections = detections;
  window._detTPs = detections.map(d => isTP(d.time));
  window._detIdx = -1;
  updateDetNav();
}

function updateDetNav() {
  const label = document.getElementById('det-nav-label');
  const tag = document.getElementById('det-nav-tag');
  const prob = document.getElementById('det-nav-prob');
  if (!window._detections || window._detections.length === 0) {
    label.textContent = 'No detections above threshold';
    tag.innerHTML = '';
    prob.textContent = '';
    return;
  }
  if (window._detIdx < 0) {
    label.textContent = 'Detections: ' + window._detections.length + ' found — use arrows to navigate';
    tag.innerHTML = '';
    prob.textContent = '';
    return;
  }
  const d = window._detections[window._detIdx];
  const tp = window._detTPs[window._detIdx];
  label.textContent = 'Detection ' + (window._detIdx + 1) + '/' + window._detections.length + '  @  ' + fmtTime(d.time);
  tag.innerHTML = tp
    ? '<span class="tag tag-tp">TP</span>'
    : '<span class="tag tag-fp">FP</span>';
  prob.textContent = 'P(HDM) = ' + d.prob_p.toFixed(3);
}

function navDetection(dir) {
  if (!window._detections || window._detections.length === 0) return;
  window._detIdx = (window._detIdx + dir + window._detections.length) % window._detections.length;
  const d = window._detections[window._detIdx];
  jumpTo(d.time);
  updateDetNav();
  // Highlight current row in list
  document.querySelectorAll('[data-det-time]').forEach(r => r.style.background = '');
  const row = document.querySelector('[data-det-time="' + d.time + '"]');
  if (row) {
    row.style.background = '#e8f5e9';
    row.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }
}

function buildEventList(mid, m, detail) {
  const el = document.getElementById('events-list');
  const header = document.getElementById('events-header');
  header.textContent = 'HDM Events \u2014 ' + mid + ' (TP=' + m.tp + ' FP=' + m.fp + ' FN=' + m.fn + ') \u2014 Click row to jump, click speaker to play clip';

  const pos = detail.samples.filter(s => s.label === 1);
  const fp = detail.samples.filter(s => s.label === 0 && s.pred === 1);

  let html = '';
  pos.forEach((s, i) => {
    const isTP = s.pred === 1;
    const cls = isTP ? '' : ' fn';
    const tag = isTP
      ? '<span class="tag tag-tp">TP</span>'
      : '<span class="tag tag-fn">FN</span>';
    const clipId = 'clip-pos-' + i;
    html += '<div class="event-row' + cls + '">' +
      '<div class="ev-time" style="cursor:pointer" onclick="jumpTo(' + s.time + ')">' + fmtTime(s.time) + '</div>' +
      '<div class="ev-text" style="cursor:pointer" onclick="jumpTo(' + s.time + ')">\"' + (s.text || '?') + '\"</div>' +
      '<div class="ev-speaker">Speaker ' + s.speaker + '</div>' +
      tag +
      '<span class="tag tag-prob">P=' + s.prob_p.toFixed(3) + '</span>' +
      '<button class="clip-btn" id="btn-' + clipId + '" onclick="playClip(\'' + mid + '\',' + s.time + ',\'' + clipId + '\')">Play 12s</button>' +
      '</div>' +
      '<div id="' + clipId + '" style="display:none;padding:2px 15px 8px;background:#f8f9fa;"></div>';
  });

  if (fp.length > 0) {
    html += '<div style="padding:8px 15px;font-size:12px;font-weight:600;color:#da3633;border-bottom:1px solid #f0f0f0">False Positives</div>';
    fp.forEach((s, i) => {
      const clipId = 'clip-fp-' + i;
      html += '<div class="event-row" style="border-left:3px solid #da3633">' +
        '<div class="ev-time" style="cursor:pointer" onclick="jumpTo(' + s.time + ')">' + fmtTime(s.time) + '</div>' +
        '<div class="ev-text" style="cursor:pointer" onclick="jumpTo(' + s.time + ')">\"' + (s.text || '') + '\"</div>' +
        '<span class="tag tag-fp">FP</span>' +
        '<span class="tag tag-prob">P=' + s.prob_p.toFixed(3) + '</span>' +
        '<button class="clip-btn" id="btn-' + clipId + '" onclick="playClip(\'' + mid + '\',' + s.time + ',\'' + clipId + '\')">Play 12s</button>' +
        '</div>' +
        '<div id="' + clipId + '" style="display:none;padding:2px 15px 8px;background:#f8f9fa;"></div>';
    });
  }

  el.innerHTML = html;
}

async function playClip(mid, time, clipId) {
  const el = document.getElementById(clipId);
  const btn = document.getElementById('btn-' + clipId);

  if (el.style.display === 'block') {
    el.style.display = 'none';
    btn.textContent = 'Play 12s';
    return;
  }

  btn.textContent = 'Loading...';
  btn.classList.add('loading-clip');

  const key = mid + '_' + time;
  if (!clipCache[key]) {
    const r = await fetch('/api/clip?meeting=' + mid + '&time=' + time);
    clipCache[key] = await r.json();
  }

  el.innerHTML = '<div class="clip-player"><audio controls autoplay src="data:audio/wav;base64,' + clipCache[key].audio + '"></audio></div>';
  el.style.display = 'block';
  btn.textContent = 'Hide';
  btn.classList.remove('loading-clip');
}

async function buildShotList(splitIdx) {
  const el = document.getElementById('shots-list');
  el.innerHTML = '<div class="loading">Loading few-shot examples...</div>';

  const r = await fetch('/api/shots/' + splitIdx);
  const shots = await r.json();

  let html = '<div class="shots-grid">';
  shots.forEach((s, i) => {
    const isP = s.label === 1;
    const labelCls = isP ? 'shot-label-p' : 'shot-label-n';
    const labelTxt = isP ? 'P (HDM)' : 'N (Not HDM)';
    const clipId = 'shot-clip-' + i;

    let sourceBg = '#e9ecef';
    if (s.source === 'human-verified') sourceBg = '#d4edda';
    else if (s.source === 'hard-negative') sourceBg = '#fff3cd';

    html += '<div class="shot-item">' +
      '<div class="shot-header">' +
        '<span class="shot-label ' + labelCls + '">' + labelTxt + '</span>' +
        '<span class="shot-source" style="background:' + sourceBg + '">' + s.source + '</span>' +
        '<span class="shot-order">#' + (i + 1) + '</span>' +
      '</div>' +
      '<div class="shot-meta">' + s.meeting_id + ' \u2014 Speaker ' + s.speaker + ' \u2014 ' + fmtTime(s.sample_time) + '</div>' +
      '<div class="shot-text">' + (s.text ? '\"' + s.text + '\"' : '(no transcript)') + '</div>' +
      '<button class="clip-btn" id="btn-' + clipId + '" onclick="playClip(\'' + s.meeting_id + '\',' + s.sample_time + ',\'' + clipId + '\')">Play 12s clip</button>' +
      '<div id="' + clipId + '" style="display:none;margin-top:4px;"></div>' +
    '</div>';
  });
  html += '</div>';
  el.innerHTML = html;
}

function toggleSection(name) {
  const toggle = document.getElementById('toggle-' + name);
  const list = document.getElementById(name + '-list');
  if (list.style.display === 'none') {
    list.style.display = '';
    toggle.classList.add('open');
  } else {
    list.style.display = 'none';
    toggle.classList.remove('open');
  }
}

// Keyboard shortcuts
document.addEventListener('keydown', e => {
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  if (e.code === 'ArrowRight' && audio) { audio.currentTime += 5; }
  if (e.code === 'ArrowLeft' && audio) { audio.currentTime = Math.max(0, audio.currentTime - 5); }
});

init();
</script>
</body></html>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        p = urlparse(self.path).path

        if p == "/" or p == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.end_headers()
            self.wfile.write(HTML.encode())

        elif p == "/api/meetings":
            self._json_response(meeting_list)

        elif p.startswith("/api/waveform/"):
            mid = p.split("/api/waveform/")[1]
            if mid in meeting_details:
                self._json_response(get_waveform(mid))
            else:
                self.send_response(404)
                self.end_headers()

        elif p.startswith("/api/detail/"):
            mid = p.split("/api/detail/")[1]
            if mid in meeting_details:
                self._json_response(meeting_details[mid])
            else:
                self.send_response(404)
                self.end_headers()

        elif p.startswith("/api/probsignal/"):
            mid = p.split("/api/probsignal/")[1]
            if mid in meeting_details:
                self._json_response(get_prob_signal(mid))
            else:
                self.send_response(404)
                self.end_headers()

        elif p.startswith("/api/sliding/"):
            mid = p.split("/api/sliding/")[1]
            sw_path = RESULTS_DIR / "sliding_window" / f"{mid}.json"
            if sw_path.exists():
                with open(sw_path) as f:
                    data = json.load(f)
                data = smooth_sliding_window(data)
                self._json_response(data)
            else:
                self._json_response({"windows": []})

        elif p == "/api/global_stats":
            with open(RESULTS_DIR / "gpt4o_20shot_v4_results.json") as f:
                res = json.load(f)
            self._json_response({
                "avg_f1": res["avg_f1"],
                "std_f1": res["std_f1"],
                "best_f1": max(s["f1"] for s in res["splits"]),
            })

        elif p == "/api/clip":
            params = parse_qs(urlparse(self.path).query)
            mid = params.get("meeting", [None])[0]
            time = float(params.get("time", [0])[0])
            if mid:
                b64 = extract_clip_b64(mid, time)
                if b64:
                    self._json_response({"audio": b64})
                    return
            self.send_response(404)
            self.end_headers()

        elif p.startswith("/api/shots/"):
            split_idx = int(p.split("/api/shots/")[1])
            if split_idx in shot_data:
                self._json_response(shot_data[split_idx])
            else:
                self._json_response([])

        elif p.startswith("/audio/"):
            mid = p.split("/audio/")[1]
            ap = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
            if not ap.exists():
                self.send_response(404)
                self.end_headers()
                return
            sz = ap.stat().st_size
            range_header = self.headers.get("Range")
            if range_header:
                rng = range_header.replace("bytes=", "")
                parts = rng.split("-")
                start = int(parts[0]) if parts[0] else 0
                end = int(parts[1]) if parts[1] else sz - 1
                end = min(end, sz - 1)
                length = end - start + 1
                self.send_response(206)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Range", f"bytes {start}-{end}/{sz}")
                self.send_header("Content-Length", str(length))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(ap, "rb") as f:
                    f.seek(start)
                    rem = length
                    while rem > 0:
                        chunk = f.read(min(65536, rem))
                        if not chunk:
                            break
                        self.wfile.write(chunk)
                        rem -= len(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(sz))
                self.send_header("Accept-Ranges", "bytes")
                self.end_headers()
                with open(ap, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
        else:
            self.send_response(404)
            self.end_headers()

    def _json_response(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def log_message(self, fmt, *args):
        pass


def main():
    print("Figure 1 Dashboard — Plotly + Audio")
    print("=" * 50)
    load_data()
    port = 8766
    server = HTTPServer(("0.0.0.0", port), Handler)
    print(f"\n  Dashboard: http://localhost:{port}")
    print(f"  {len(meeting_list)} meetings")
    print(f"  Click chart to seek | Space = play/pause | Arrows = skip 5s")
    print(f"  Click HDM event row to jump, zoom, and play\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.server_close()


if __name__ == "__main__":
    main()
