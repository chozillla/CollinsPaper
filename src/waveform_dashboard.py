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
from scipy.integrate import trapezoid

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
meeting_labels = {}  # mid -> list of {idx, time, text, speaker, label}


def smooth_sliding_window(data, sigma=0.5):
    """Gaussian-smooth the prob_p signal to match the paper's smooth curve."""
    windows = data.get("windows", [])
    if len(windows) < 5:
        return data
    probs = np.array([w["prob_p"] for w in windows])
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


def compute_alignment_score(sw_path, hdm_regions):
    """Compute alignment score between model signal and ground truth HDMs."""
    with open(sw_path) as f:
        data = json.load(f)
    windows = data["windows"]
    times = np.array([w["time"] for w in windows])
    probs = np.array([w["prob_p"] for w in windows])
    duration = data["duration"]

    gt = np.zeros_like(probs)
    for h in hdm_regions:
        mask = (times >= h["start"] - 4.0) & (times <= h["end"] + 4.0)
        gt[mask] = 1.0

    nabc = trapezoid(np.abs(probs - gt), times) / duration

    peaks_at_hdm = []
    for h in hdm_regions:
        nearby = probs[(times >= h["start"] - 6.0) & (times <= h["end"] + 6.0)]
        if len(nearby) > 0:
            peaks_at_hdm.append(float(nearby.max()))
    peak_at_hdm = np.mean(peaks_at_hdm) if peaks_at_hdm else 0.0

    non_hdm_probs = probs[gt == 0]
    noise_floor = float(non_hdm_probs.mean()) if len(non_hdm_probs) > 0 else 0.0
    false_alarm_rate = float((non_hdm_probs > 0.5).sum()) / max(len(non_hdm_probs), 1) if len(non_hdm_probs) > 0 else 0.0

    detection_score = peak_at_hdm
    specificity_score = 1.0 - min(noise_floor / 0.5, 1.0)
    alignment_score = round(100.0 * (0.5 * detection_score + 0.5 * specificity_score), 1)

    return {
        "alignment_score": alignment_score,
        "nabc": round(nabc, 4),
        "peak_at_hdm": round(peak_at_hdm, 3),
        "noise_floor": round(noise_floor, 3),
        "false_alarm_rate": round(false_alarm_rate * 100, 1),
    }


def load_data():
    global meeting_list, meeting_details, shot_data

    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)
    human_labels = {}
    p = ROOT / "data" / "hdm_labels.json"
    if p.exists():
        with open(p) as f:
            human_labels = json.load(f)

    md = defaultdict(lambda: {"hdm_regions": []})

    # Build per-meeting labeled items for navigation
    global meeting_labels
    meeting_labels = defaultdict(list)
    for idx_str, lbl in human_labels.items():
        idx = int(idx_str)
        if idx < len(meta["positive"]):
            ex = meta["positive"][idx]
            meeting_labels[ex["meeting_id"]].append({
                "idx": idx,
                "time": round(ex["sample_time"], 3),
                "text": ex.get("text", ""),
                "speaker": ex.get("speaker", ""),
                "label": lbl,
            })
    for mid in meeting_labels:
        meeting_labels[mid].sort(key=lambda x: x["time"])

    # Build ground truth HDM regions from dataset metadata
    for i, ex in enumerate(meta["positive"]):
        mid = ex["meeting_id"]
        if ex.get("hdm_start") is not None:
            md[mid]["hdm_regions"].append({
                "start": round(ex["hdm_start"], 3),
                "end": round(ex["hdm_end"], 3),
                "text": ex.get("text", ""),
                "speaker": ex.get("speaker", ""),
            })

    for mid, data in sorted(md.items()):
        ap = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        if not ap.exists():
            continue
        # Only show meetings with 10-shot sliding window results
        sw_path = RESULTS_DIR / "sliding_window_10shot" / f"{mid}.json"
        if not sw_path.exists():
            continue
        n_hdms = len(data["hdm_regions"])
        if n_hdms == 0:
            continue
        info = sf.info(str(ap))
        data["hdm_regions"].sort(key=lambda x: x["start"])

        # Compute alignment score
        score_info = compute_alignment_score(sw_path, data["hdm_regions"])

        # AMI corpus metadata from meeting ID
        site_map = {"E": "Edinburgh", "I": "Idiap", "T": "TNO"}
        session_map = {"a": "Kick-off", "b": "Functional", "c": "Conceptual", "d": "Detailed"}
        site = site_map.get(mid[0], mid[0])
        group = mid[:-1]
        session = mid[-1]
        phase = session_map.get(session, session)

        meeting_list.append({
            "id": mid, "duration": round(info.duration, 2),
            "n_pos": n_hdms,
            "site": site,
            "group": group,
            "session": session,
            "phase": phase,
            **score_info,
        })
        meeting_details[mid] = data

    meeting_list.sort(key=lambda x: -x["alignment_score"])
    print(f"Loaded {len(meeting_list)} meetings (Gemini 10-shot)")


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

.sort-bar { display: flex; gap: 6px; padding: 8px 20px; justify-content: center; align-items: center; background: #fff; border-bottom: 1px solid #e0e0e0; }
.sort-bar label { font-size: 11px; color: #888; font-weight: 600; }
.sort-btn { padding: 4px 12px; background: #f8f9fa; border: 1px solid #dee2e6; color: #495057; border-radius: 5px; cursor: pointer; font-size: 11px; transition: all 0.15s; }
.sort-btn:hover { background: #e9ecef; }
.sort-btn.active { background: #2c3e50; border-color: #2c3e50; color: #fff; }

.nav-bar { display: flex; gap: 4px; padding: 10px 20px; flex-wrap: wrap; justify-content: center; background: #fff; border-bottom: 1px solid #e0e0e0; }
.nav-btn { padding: 5px 10px; background: #f8f9fa; border: 1px solid #dee2e6; color: #495057; border-radius: 5px; cursor: pointer; font-size: 11px; transition: all 0.15s; display: flex; flex-direction: column; align-items: center; gap: 1px; min-width: 80px; }
.nav-btn:hover { background: #e9ecef; }
.nav-btn.active { background: #1a9641; border-color: #1a9641; color: #fff; }
.nav-btn .nav-score { font-size: 9px; font-weight: 700; }
.nav-btn .nav-score.excellent { color: #1a9641; }
.nav-btn .nav-score.good { color: #3498db; }
.nav-btn .nav-score.fair { color: #f39c12; }
.nav-btn .nav-score.poor { color: #e74c3c; }
.nav-btn.active .nav-score { color: #fff; }

.content { max-width: 1200px; margin: 15px auto; padding: 0 15px; }

.chart-card { background: #fff; border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.chart-card .chart-header { padding: 10px 15px; border-bottom: 1px solid #f0f0f0; display: flex; justify-content: space-between; align-items: center; }
.chart-card .chart-title { font-size: 14px; font-weight: 600; color: #2c3e50; }
.chart-card .chart-meta { font-size: 11px; color: #999; }
#chart { width: 100%; height: 550px; }

#chart-wrapper { position: relative; }
#playback-cursor {
  position: absolute; top: 0; bottom: 0; width: 3px;
  background: rgba(231,76,60,0.9); pointer-events: none;
  z-index: 10; display: none;
  box-shadow: 0 0 6px rgba(231,76,60,0.4);
}
#playback-cursor::after {
  content: ''; position: absolute; top: 50%; left: 50%;
  transform: translate(-50%,-50%);
  width: 9px; height: 9px; border-radius: 50%;
  background: #e74c3c; border: 2px solid #fff;
  box-shadow: 0 0 4px rgba(0,0,0,0.3);
}

/* --- Transport Controls --- */
.transport {
  background: #f8f9fa; padding: 0; margin: 0;
  border-top: none;
  border-radius: 0 0 10px 10px;
}
.transport-inner {
  display: flex; align-items: center; justify-content: center; gap: 10px;
  padding: 8px 15px 10px;
}
.ctrl-btn {
  width: 32px; height: 32px; border-radius: 50%; border: none; cursor: pointer;
  display: inline-flex; align-items: center; justify-content: center;
  background: #e9ecef; color: #495057; font-size: 12px; font-weight: 700;
  transition: all 0.12s; flex-shrink: 0;
}
.ctrl-btn:hover { background: #dee2e6; }
.ctrl-btn.primary {
  width: 40px; height: 40px; background: #1a9641; color: #fff; font-size: 17px;
}
.ctrl-btn.primary:hover { background: #158a38; }
.ctrl-btn.primary.playing { background: #e74c3c; }
.ctrl-btn.primary.playing:hover { background: #c0392b; }

.time-display {
  font-family: 'SF Mono', 'Fira Code', monospace; font-size: 12px; color: #555;
  white-space: nowrap; min-width: 100px; text-align: center;
}
.time-elapsed { color: #2c3e50; font-weight: 600; }
.time-remaining { color: #999; font-size: 11px; }
.speed-select {
  padding: 3px 6px; border-radius: 5px; border: 1px solid #dee2e6;
  font-size: 11px; font-weight: 600; color: #495057; background: #f8f9fa;
  cursor: pointer; flex-shrink: 0;
}
.speed-select:hover { border-color: #adb5bd; }
.kbd-hints {
  font-size: 9px; color: #bbb;
  display: flex; gap: 6px; align-items: center; flex-shrink: 0;
}
.kbd { display: inline-block; padding: 1px 4px; background: #e9ecef; border-radius: 3px;
       font-family: 'SF Mono', monospace; font-size: 8px; color: #666; border: 1px solid #dee2e6; }

.audio-card { background: #fff; border: 1px solid #e0e0e0; border-radius: 10px; padding: 12px 15px; margin-bottom: 12px; display: flex; align-items: center; gap: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.06); }
.play-btn { padding: 8px 20px; background: #1a9641; color: #fff; border: none; border-radius: 6px; cursor: pointer; font-size: 13px; font-weight: 600; min-width: 70px; }
.play-btn:hover { background: #158a38; }
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
  <p>Replicating Collins et al. Figure 1 — Gemini 2.5 Flash (10-shot) on AMI Corpus | Click chart to seek audio</p>
  <p style="font-size:10px;color:#999;margin-top:2px;">Green line = P(HDM) from sliding window. Red bands = ground truth HDM events.</p>
</div>

<div class="stats-bar">
  <div class="stat-card"><div class="val" id="s-model">-</div><div class="lbl">Model</div></div>
  <div class="stat-card"><div class="val muted">0.87</div><div class="lbl">Paper F1</div></div>
  <div class="stat-card"><div class="val" id="s-meetings">-</div><div class="lbl">Meetings</div></div>
  <div class="stat-card"><div class="val" id="s-hdms">-</div><div class="lbl">HDMs</div></div>
  <div class="stat-card"><div class="val" id="s-avg-score">-</div><div class="lbl">Avg Score</div></div>
  <div class="stat-card"><div class="val" id="s-detected">-</div><div class="lbl">Detected</div></div>
</div>

<div class="sort-bar">
  <label>Sort:</label>
  <button class="sort-btn active" id="sort-score" onclick="sortMeetings('score')">Score</button>
  <button class="sort-btn" id="sort-name" onclick="sortMeetings('name')">Name</button>
  <button class="sort-btn" id="sort-hdms" onclick="sortMeetings('hdms')">HDMs</button>
  <button class="sort-btn" id="sort-duration" onclick="sortMeetings('duration')">Duration</button>
  <span style="margin:0 6px;color:#ccc;">|</span>
  <label>Filter:</label>
  <select class="sort-btn" id="filter-site" onchange="filterMeetings()" style="padding:4px 8px;cursor:pointer;">
    <option value="">All Sites</option>
    <option value="Edinburgh">Edinburgh</option>
    <option value="Idiap">Idiap</option>
    <option value="TNO">TNO</option>
  </select>
  <select class="sort-btn" id="filter-phase" onchange="filterMeetings()" style="padding:4px 8px;cursor:pointer;">
    <option value="">All Phases</option>
    <option value="a">a — Kick-off</option>
    <option value="b">b — Functional</option>
    <option value="c">c — Conceptual</option>
    <option value="d">d — Detailed</option>
  </select>
  <select class="sort-btn" id="filter-group" onchange="filterMeetings()" style="padding:4px 8px;cursor:pointer;">
    <option value="">All Groups</option>
  </select>
  <span style="margin-left:8px;font-size:10px;color:#aaa;" id="sort-info"></span>
</div>
<div class="nav-bar" id="nav"></div>

<div class="content">
  <div class="chart-card">
    <div class="chart-header">
      <div class="chart-title" id="chart-title">Select a meeting</div>
      <div class="chart-meta" id="chart-meta"></div>
    </div>
    <div id="chart-wrapper" style="position:relative;">
      <div id="chart"><div class="loading">Loading...</div></div>
      <div id="playback-cursor"></div>
    </div>
    <div class="transport">
      <div class="transport-inner">
        <button class="ctrl-btn" onclick="skipAudio(-10)" title="Back 10s">&#9668;&#9668;</button>
        <button class="ctrl-btn" onclick="skipAudio(-5)" title="Back 5s">&#9668;</button>
        <button class="ctrl-btn primary" id="play-btn" onclick="togglePlay()" title="Play/Pause">&#9654;</button>
        <button class="ctrl-btn" onclick="skipAudio(5)" title="Forward 5s">&#9658;</button>
        <button class="ctrl-btn" onclick="skipAudio(10)" title="Forward 10s">&#9658;&#9658;</button>
        <audio id="audio" preload="none"></audio>
        <div class="time-display" id="time-display">
          <span class="time-elapsed">0:00</span> <span class="time-remaining">/ 0:00</span>
        </div>
        <select class="speed-select" id="speed" onchange="audio.playbackRate=parseFloat(this.value)">
          <option value="0.5">0.5x</option>
          <option value="0.75">0.75x</option>
          <option value="1" selected>1.0x</option>
          <option value="1.25">1.25x</option>
          <option value="1.5">1.5x</option>
          <option value="2">2.0x</option>
        </select>
        <div class="kbd-hints">
          <span><kbd class="kbd">Space</kbd></span>
          <span><kbd class="kbd">&larr;&rarr;</kbd></span>
          <span><kbd class="kbd">J K L</kbd></span>
        </div>
      </div>
    </div>
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

  <div class="audio-card" style="background:#fff8f0;border-color:#ffe0b2;padding:10px 15px;">
    <div style="display:flex;align-items:center;gap:10px;width:100%;">
      <button class="play-btn" style="background:#e65100;min-width:36px;padding:6px 12px;font-size:12px;border-radius:5px;" onclick="navLabel(-1)">&#9664;</button>
      <button class="play-btn" style="background:#e65100;min-width:36px;padding:6px 12px;font-size:12px;border-radius:5px;" onclick="navLabel(1)">&#9654;</button>
      <div style="font-size:13px;font-weight:600;color:#2c3e50;" id="lbl-nav-label">Labels: loading...</div>
      <div style="margin-left:auto;display:flex;align-items:center;gap:6px;">
        <span id="lbl-nav-tag" style="font-size:11px;"></span>
        <span id="lbl-nav-text" style="font-size:12px;color:#666;font-style:italic;"></span>
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
let allMeetings = [];  // unfiltered master list
let meetings = [];     // filtered + sorted view
let currentMid = null;
let audio = document.getElementById('audio');
let cursorInterval = null;
let clipCache = {};
let currentSort = 'score';

function fmtTime(s) {
  if (!s || isNaN(s)) return '0:00';
  const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}

function scoreGrade(s) {
  if (s >= 80) return 'excellent';
  if (s >= 60) return 'good';
  if (s >= 40) return 'fair';
  return 'poor';
}

function siteTag(site) {
  const colors = {Edinburgh: '#2980b9', Idiap: '#8e44ad', TNO: '#e67e22'};
  return '<span style="font-size:8px;padding:1px 4px;border-radius:3px;color:#fff;background:' +
    (colors[site]||'#888') + '">' + (site||'?').substring(0,3) + '</span>';
}

function renderNav() {
  const nav = document.getElementById('nav');
  nav.innerHTML = '';
  meetings.forEach((m) => {
    const btn = document.createElement('button');
    btn.className = 'nav-btn' + (m.id === currentMid ? ' active' : '');
    const score = m.alignment_score || 0;
    const grade = scoreGrade(score);
    btn.innerHTML = '<span style="display:flex;align-items:center;gap:3px;">' + siteTag(m.site) +
      ' ' + m.id + '</span><span class="nav-score ' + grade + '">' + score.toFixed(0) +
      ' <span style="font-weight:400;opacity:0.7">' + (m.phase||'') + '</span></span>';
    btn.onclick = () => selectMeeting(m.id);
    btn.id = 'btn-' + m.id;
    nav.appendChild(btn);
  });
  const scores = meetings.map(m => m.alignment_score || 0);
  const avg = scores.length ? (scores.reduce((a,b) => a+b, 0) / scores.length) : 0;
  document.getElementById('sort-info').textContent =
    meetings.length + '/' + allMeetings.length + ' meetings | avg score: ' + avg.toFixed(1);
}

function applySort() {
  if (currentSort === 'score') meetings.sort((a, b) => (b.alignment_score||0) - (a.alignment_score||0));
  else if (currentSort === 'name') meetings.sort((a, b) => a.id.localeCompare(b.id));
  else if (currentSort === 'hdms') meetings.sort((a, b) => b.n_pos - a.n_pos);
  else if (currentSort === 'duration') meetings.sort((a, b) => b.duration - a.duration);
}

function sortMeetings(by) {
  currentSort = by;
  document.querySelectorAll('.sort-bar .sort-btn').forEach(b => {
    if (b.id && b.id.startsWith('sort-')) b.classList.remove('active');
  });
  document.getElementById('sort-' + by).classList.add('active');
  applySort();
  renderNav();
}

function filterMeetings() {
  const site = document.getElementById('filter-site').value;
  const phase = document.getElementById('filter-phase').value;
  const group = document.getElementById('filter-group').value;
  meetings = allMeetings.filter(m => {
    if (site && m.site !== site) return false;
    if (phase && m.session !== phase) return false;
    if (group && m.group !== group) return false;
    return true;
  });
  applySort();
  renderNav();
}

function populateGroupFilter() {
  const sel = document.getElementById('filter-group');
  const groups = [...new Set(allMeetings.map(m => m.group))].sort();
  groups.forEach(g => {
    const count = allMeetings.filter(m => m.group === g).length;
    const opt = document.createElement('option');
    opt.value = g;
    opt.textContent = g + ' (' + count + ' sessions)';
    sel.appendChild(opt);
  });
}

async function init() {
  const r = await fetch('/api/meetings');
  allMeetings = await r.json();
  meetings = [...allMeetings];
  populateGroupFilter();
  applySort();
  renderNav();
  // Compute global stats
  const resp2 = await fetch('/api/global_stats');
  const gs = await resp2.json();
  document.getElementById('s-model').textContent = '10-shot';
  document.getElementById('s-model').style.fontSize = '16px';
  document.getElementById('s-meetings').textContent = gs.n_meetings;
  document.getElementById('s-hdms').textContent = meetings.reduce((a, m) => a + m.n_pos, 0);
  const scores = meetings.map(m => m.alignment_score || 0);
  const avgScore = scores.reduce((a,b) => a+b, 0) / scores.length;
  document.getElementById('s-avg-score').textContent = avgScore.toFixed(0);
  const detected = meetings.filter(m => (m.peak_at_hdm||0) > 0.5).length;
  document.getElementById('s-detected').textContent = detected + '/' + meetings.length;

  if (meetings.length > 0) selectMeeting(meetings[0].id);
}

async function selectMeeting(mid) {
  if (currentMid === mid) return;
  currentMid = mid;
  const m = meetings.find(x => x.id === mid);

  // Update nav
  document.querySelectorAll('.nav-btn').forEach(b => {
    b.classList.remove('active');
    // Restore score color when deactivated
    const sc = b.querySelector('.nav-score');
    if (sc) { const g = sc.getAttribute('data-grade'); if (g) sc.className = 'nav-score ' + g; }
  });
  const activeBtn = document.getElementById('btn-' + mid);
  activeBtn.classList.add('active');
  const activeSc = activeBtn.querySelector('.nav-score');
  if (activeSc) { activeSc.setAttribute('data-grade', activeSc.className.replace('nav-score ','')); activeSc.className = 'nav-score'; }

  // Update stats
  const score = m.alignment_score || 0;
  const grade = scoreGrade(score);
  const gradeLabel = grade.charAt(0).toUpperCase() + grade.slice(1);
  document.getElementById('chart-title').textContent = 'Audio Waveform vs. Model Prediction and Ground Truth \u2014 ' + mid;
  document.getElementById('chart-meta').textContent =
    'Score: ' + score.toFixed(0) + ' (' + gradeLabel + ') | ' +
    (m.site||'') + ' | Group ' + (m.group||'') + ' | Session ' + (m.session||'').toUpperCase() + ' (' + (m.phase||'') + ') | ' +
    (m.duration / 60).toFixed(1) + ' min | ' + m.n_pos + ' HDMs | ' +
    'Peak: ' + (m.peak_at_hdm||0).toFixed(2) + ' | Noise: ' + (m.noise_floor||0).toFixed(3);

  // Load audio
  audio.pause();
  audio.src = '/audio/' + mid;
  audio.load();
  document.getElementById('play-btn').innerHTML = '&#9654;';
  document.getElementById('play-btn').classList.remove('playing');

  // Show loading
  document.getElementById('chart').innerHTML = '<div class="loading">Loading waveform...</div>';

  // Fetch waveform + meeting data + sliding window + labels in parallel
  const [waveResp, detailResp, slidingResp, labelsResp] = await Promise.all([
    fetch('/api/waveform/' + mid),
    fetch('/api/detail/' + mid),
    fetch('/api/sliding/' + mid),
    fetch('/api/labels/' + mid),
  ]);
  const wave = await waveResp.json();
  const detail = await detailResp.json();
  const sliding = await slidingResp.json();
  const labels = await labelsResp.json();

  buildChart(mid, m, wave, detail, sliding);
  buildDetectionsList(mid, detail, sliding);
  buildLabelNav(labels);
  buildEventList(mid, m, detail);

  // Start cursor update
  if (cursorInterval) clearInterval(cursorInterval);
  cursorInterval = setInterval(updateCursor, 50);
}

function buildChart(mid, m, wave, detail, sliding) {
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
    fill: 'tonexty', fillcolor: 'rgba(100,149,237,0.2)',
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
      type: 'scatter', mode: 'lines+markers',
      line: { color: '#006400', width: 1.8, shape: 'linear' },
      marker: { color: '#006400', size: 2.5 },
      name: 'Model Prediction',
      text: swTexts, hoverinfo: 'text',
      yaxis: 'y2',
      legendrank: 2,
    });
  }

  // 3. Dummy traces for legend
  traces.push({
    x: [null], y: [null], type: 'scatter', mode: 'lines',
    line: { color: 'rgba(255,50,50,0.7)', width: 10 },
    name: 'Ground Truth Event',
    legendrank: 1,
  });
  traces.push({
    x: [null], y: [null], type: 'scatter', mode: 'lines',
    line: { color: 'orange', width: 2, dash: 'dash' },
    name: 'Decision Threshold (0.97)',
    legendrank: 3,
  });
  traces.push({
    x: [null], y: [null], type: 'scatter', mode: 'lines',
    line: { color: 'rgba(0,200,83,0.5)', width: 10 },
    name: 'Positive Prediction Region',
    legendrank: 4,
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

  // Positive prediction regions (bright green shading where prob >= threshold)
  if (hasSliding) {
    let regionStart = null;
    const step = sliding.step_s || 4;
    const THRESHOLD = 0.97;
    sliding.windows.forEach((w, i) => {
      if (w.prob_p >= THRESHOLD && regionStart === null) {
        regionStart = w.time - step / 2;
      } else if (w.prob_p < THRESHOLD && regionStart !== null) {
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
      title: { text: 'Amplitude', font: { size: 11, color: 'steelblue' } },
      tickfont: { size: 9, color: 'steelblue' },
      showgrid: true, gridcolor: 'rgba(0,0,0,0.05)',
      fixedrange: true,
    },
    yaxis2: {
      title: { text: 'Model Probability', font: { size: 11, color: '#006400' } },
      tickfont: { size: 9, color: '#006400' },
      overlaying: 'y', side: 'right',
      range: [-0.05, 1.1],
      showgrid: false,
      fixedrange: true,
    },
    shapes: shapes,
    annotations: annotations,
    legend: {
      orientation: 'h', x: 0.5, xanchor: 'center', y: 1.15,
      font: { size: 11 },
      itemwidth: 40,
    },
    margin: { l: 60, r: 60, t: 55, b: 45 },
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

function positionCursor() {
  const cursor = document.getElementById('playback-cursor');
  const chartEl = document.getElementById('chart');
  if (!cursor || !chartEl || !chartEl._fullLayout || !audio) return;
  const xaxis = chartEl._fullLayout.xaxis;
  if (!xaxis) return;
  const t = audio.currentTime;
  const xRange = xaxis.range;
  const frac = (t - xRange[0]) / (xRange[1] - xRange[0]);
  if (frac < 0 || frac > 1) {
    cursor.style.display = 'none';
    return;
  }
  const px = xaxis._offset + frac * xaxis._length;
  cursor.style.display = 'block';
  cursor.style.left = px + 'px';
}

function updateCursor() {
  if (!audio || !audio.duration || !currentMid) return;

  // Position the CSS cursor overlay (fast, no Plotly re-render)
  positionCursor();

  // Update time display
  const elapsed = fmtTime(audio.currentTime);
  const remaining = '-' + fmtTime(audio.duration - audio.currentTime);
  document.getElementById('time-display').innerHTML =
    '<span class="time-elapsed">' + elapsed + '</span> <span class="time-remaining">' + remaining + '</span>';

  // Update play button state
  const pb = document.getElementById('play-btn');
  if (audio.paused) {
    pb.innerHTML = '&#9654;';
    pb.classList.remove('playing');
  } else {
    pb.innerHTML = '&#9646;&#9646;';
    pb.classList.add('playing');
  }

  // Auto-scroll: if zoomed in and cursor goes past visible window
  const chartEl = document.getElementById('chart');
  if (chartEl && chartEl.layout && chartEl.layout.xaxis && chartEl.layout.xaxis.range) {
    const xRange = chartEl.layout.xaxis.range;
    const viewStart = xRange[0];
    const viewEnd = xRange[1];
    const viewDur = viewEnd - viewStart;
    const t = audio.currentTime;
    const m = meetings.find(x => x.id === currentMid);
    const fullDur = m ? m.duration : audio.duration;

    if (viewDur < fullDur * 0.9 && !audio.paused) {
      if (t > viewStart + viewDur * 0.75) {
        const newStart = t - viewDur * 0.25;
        const newEnd = newStart + viewDur;
        Plotly.relayout('chart', {
          'xaxis.range': [Math.max(0, newStart), Math.min(fullDur, newEnd)]
        });
      }
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

// Re-position cursor when chart is zoomed/panned
(function() {
  setTimeout(() => {
    const chartEl = document.getElementById('chart');
    if (chartEl) {
      chartEl.on('plotly_relayout', () => positionCursor());
    }
  }, 500);
})();

function skipAudio(sec) {
  if (!audio) return;
  audio.currentTime = Math.max(0, Math.min(audio.duration || 0, audio.currentTime + sec));
}

function togglePlay() {
  if (!audio || !audio.src) return;
  if (audio.paused) {
    audio.play();
  } else {
    audio.pause();
  }
  // Button state updated in updateCursor
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

function buildLabelNav(labels) {
  window._labels = labels;
  window._lblIdx = -1;
  updateLblNav();
}

function updateLblNav() {
  const label = document.getElementById('lbl-nav-label');
  const tag = document.getElementById('lbl-nav-tag');
  const text = document.getElementById('lbl-nav-text');
  if (!window._labels || window._labels.length === 0) {
    label.textContent = 'No labels for this meeting';
    tag.innerHTML = '';
    text.textContent = '';
    return;
  }
  if (window._lblIdx < 0) {
    label.textContent = 'Labels: ' + window._labels.length + ' labeled — use arrows to navigate';
    tag.innerHTML = '';
    text.textContent = '';
    return;
  }
  const l = window._labels[window._lblIdx];
  label.textContent = 'Label ' + (window._lblIdx + 1) + '/' + window._labels.length + '  @  ' + fmtTime(l.time) + '  [#' + l.idx + ']';
  const isYes = l.label === 'yes';
  tag.innerHTML = isYes
    ? '<span class="tag tag-tp">HDM</span>'
    : '<span class="tag tag-fp">Not HDM</span>';
  text.textContent = '"' + l.text + '" — Speaker ' + l.speaker;
}

function navLabel(dir) {
  if (!window._labels || window._labels.length === 0) return;
  window._lblIdx = (window._lblIdx + dir + window._labels.length) % window._labels.length;
  const l = window._labels[window._lblIdx];
  jumpTo(l.time);
  updateLblNav();
}

function buildEventList(mid, m, detail) {
  const el = document.getElementById('events-list');
  const header = document.getElementById('events-header');
  const hdms = detail.hdm_regions || [];
  header.textContent = 'Ground Truth HDM Events \u2014 ' + mid + ' (' + hdms.length + ' events) \u2014 Click to jump';

  let html = '';
  hdms.forEach((h, i) => {
    const t = (h.start + h.end) / 2;
    const clipId = 'clip-hdm-' + i;
    html += '<div class="event-row">' +
      '<div class="ev-time" style="cursor:pointer" onclick="jumpTo(' + t + ')">' + fmtTime(t) + '</div>' +
      '<div class="ev-text" style="cursor:pointer" onclick="jumpTo(' + t + ')">\"' + (h.text || '?') + '\"</div>' +
      '<div class="ev-speaker">Speaker ' + h.speaker + '</div>' +
      '<span class="tag tag-tp">HDM</span>' +
      '<button class="clip-btn" id="btn-' + clipId + '" onclick="playClip(\'' + mid + '\',' + t + ',\'' + clipId + '\')">Play 12s</button>' +
      '</div>' +
      '<div id="' + clipId + '" style="display:none;padding:2px 15px 8px;background:#f8f9fa;"></div>';
  });

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
  // Ignore if user is typing in an input/select
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT') return;
  if (e.code === 'Space') { e.preventDefault(); togglePlay(); }
  if (e.code === 'ArrowRight') { e.preventDefault(); skipAudio(5); }
  if (e.code === 'ArrowLeft') { e.preventDefault(); skipAudio(-5); }
  if (e.code === 'KeyJ') { skipAudio(-10); }
  if (e.code === 'KeyL') { skipAudio(10); }
  if (e.code === 'KeyK') { if (audio && !audio.paused) audio.pause(); }
  // Number keys 1-4 for speed
  if (e.code === 'Digit1') { audio.playbackRate = 0.5; document.getElementById('speed').value = '0.5'; }
  if (e.code === 'Digit2') { audio.playbackRate = 1; document.getElementById('speed').value = '1'; }
  if (e.code === 'Digit3') { audio.playbackRate = 1.5; document.getElementById('speed').value = '1.5'; }
  if (e.code === 'Digit4') { audio.playbackRate = 2; document.getElementById('speed').value = '2'; }
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

        elif p.startswith("/api/sliding/"):
            mid = p.split("/api/sliding/")[1]
            # Prefer 10-shot results, fall back to zero-shot
            sw_path = RESULTS_DIR / "sliding_window_10shot" / f"{mid}.json"
            if not sw_path.exists():
                sw_path = RESULTS_DIR / "sliding_window" / f"{mid}.json"
            if sw_path.exists():
                with open(sw_path) as f:
                    data = json.load(f)
                data = smooth_sliding_window(data)
                self._json_response(data)
            else:
                self._json_response({"windows": []})

        elif p == "/api/global_stats":
            # Stats from Gemini 10-shot sliding window
            n_meetings = len(meeting_list)
            self._json_response({
                "n_meetings": n_meetings,
                "model": "Gemini 2.5 Flash (10-shot)",
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

        elif p.startswith("/api/labels/"):
            mid = p.split("/api/labels/")[1]
            self._json_response(meeting_labels.get(mid, []))

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
