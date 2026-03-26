"""
Interactive Figure 1 Dashboard — Plotly + Audio Playback.

Fully interactive recreation of Collins et al. Figure 1:
- Plotly chart: zoom, pan, hover for details
- Click on chart to seek audio to that timestamp
- Blue waveform, green probability line, red HDM bands, orange threshold
- Audio playback with sync cursor

Usage:
    python src/waveform_dashboard.py
    Then open http://localhost:8766
"""

import gc
import json
import numpy as np
import soundfile as sf
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from collections import defaultdict

ROOT = Path(__file__).parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
DATASET_DIR = ROOT / "data" / "dataset"
RESULTS_DIR = ROOT / "results"
SAMPLE_RATE = 16000

meeting_list = []  # [{id, duration, n_pos, tp, fp, fn}]
meeting_details = {}  # mid -> {hdm_regions, samples}
waveform_cache = {}


def load_data():
    global meeting_list, meeting_details

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
    md = defaultdict(lambda: {"hdm_regions": [], "samples": []})

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
        ti = [j for j, ex in enumerate(all_ex) if ex["meeting_id"] in tm]
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
        meeting_list.append({
            "id": mid, "duration": round(info.duration, 2),
            "n_pos": n_pos, "tp": tp, "fp": fp, "fn": fn,
        })
        meeting_details[mid] = data

    meeting_list.sort(key=lambda x: -x["n_pos"])
    print(f"Loaded {len(meeting_list)} meetings")


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
    """Build continuous probability signal at 1s resolution."""
    data = meeting_details[mid]
    ap = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
    info = sf.info(str(ap))
    duration = info.duration

    step = 1.0
    window = 4.0
    n_steps = int(duration / step) + 1
    prob = np.zeros(n_steps)

    for s in data["samples"]:
        i_start = max(0, int((s["time"] - window) / step))
        i_end = min(n_steps - 1, int(s["time"] / step))
        for i in range(i_start, i_end + 1):
            prob[i] = max(prob[i], s["prob_p"])

    return {
        "times": (np.arange(n_steps) * step).round(3).tolist(),
        "probs": prob.round(4).tolist(),
    }


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
</style>
</head>
<body>

<div class="header">
  <h1>Audio Waveform vs. Model Prediction and Ground Truth</h1>
  <p>Recreating Collins et al. Figure 1 — GPT-4o Audio v4 (20-shot) on AMI Corpus | Click chart to seek audio</p>
</div>

<div class="stats-bar">
  <div class="stat-card"><div class="val">0.97</div><div class="lbl">Our F1 (v4)</div></div>
  <div class="stat-card"><div class="val muted">0.87</div><div class="lbl">Paper F1</div></div>
  <div class="stat-card"><div class="val">1.00</div><div class="lbl">Best Split</div></div>
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

  <div class="events-card">
    <div class="events-header" id="events-header">HDM Events</div>
    <div id="events-list"></div>
  </div>
</div>

<script>
let meetings = [];
let currentMid = null;
let audio = document.getElementById('audio');
let cursorInterval = null;

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

  // Fetch waveform + meeting data + probability signal in parallel
  const [waveResp, detailResp, probResp] = await Promise.all([
    fetch('/api/waveform/' + mid),
    fetch('/api/detail/' + mid),
    fetch('/api/probsignal/' + mid),
  ]);
  const wave = await waveResp.json();
  const detail = await detailResp.json();
  const prob = await probResp.json();

  buildChart(mid, m, wave, detail, prob);
  buildEventList(mid, m, detail);

  // Start cursor update
  if (cursorInterval) clearInterval(cursorInterval);
  cursorInterval = setInterval(updateCursor, 200);
}

function buildChart(mid, m, wave, detail, prob) {
  const traces = [];

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

  // 2. Model prediction line (green, continuous)
  const hoverTexts = prob.times.map((t, i) => {
    const p = prob.probs[i];
    return 'Time: ' + fmtTime(t) + '<br>P(HDM): ' + p.toFixed(3);
  });
  traces.push({
    x: prob.times, y: prob.probs,
    type: 'scatter', mode: 'lines',
    line: { color: '#1a9641', width: 2 },
    fill: 'tozeroy', fillcolor: 'rgba(26,150,65,0.08)',
    name: 'Model Prediction',
    text: hoverTexts, hoverinfo: 'text',
    yaxis: 'y2',
  });

  // 3. Playback cursor (vertical line, updated via relayout)
  traces.push({
    x: [0, 0], y: [-1, 1],
    type: 'scatter', mode: 'lines',
    line: { color: '#e74c3c', width: 2.5 },
    name: 'Playback',
    hoverinfo: 'skip',
    yaxis: 'y',
  });

  // 4. Dummy traces for legend
  traces.push({
    x: [null], y: [null], type: 'scatter', mode: 'lines',
    line: { color: 'rgba(255,50,50,0.7)', width: 10 },
    name: 'Ground Truth Event',
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
      fillcolor: 'rgba(255,50,50,0.3)',
      line: { color: 'rgba(255,50,50,0.7)', width: 1.5 },
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

  // Positive prediction regions (green shading)
  prob.probs.forEach((p, i) => {
    if (p > 0.97 && (i === 0 || prob.probs[i - 1] <= 0.97)) {
      // Find end of region
      let j = i;
      while (j < prob.probs.length && prob.probs[j] > 0.97) j++;
      shapes.push({
        type: 'rect',
        xref: 'x', yref: 'paper',
        x0: prob.times[i], x1: prob.times[Math.min(j, prob.times.length - 1)],
        y0: 0, y1: 1,
        fillcolor: 'rgba(26,150,65,0.1)',
        line: { width: 0 },
        layer: 'below',
      });
    }
  });

  // Annotations for HDM text labels
  const annotations = detail.hdm_regions.map(r => ({
    x: (r.start + r.end) / 2,
    y: 1.02, yref: 'paper', xref: 'x',
    text: r.text || '',
    showarrow: false,
    font: { size: 9, color: '#c0392b' },
  }));

  // Add threshold label
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
      title: { text: 'Time (ms)', font: { size: 11 } },
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
      tickfont: { color: 'steelblue' },
      showgrid: true, gridcolor: 'rgba(0,0,0,0.05)',
      fixedrange: true,
    },
    yaxis2: {
      title: { text: 'Model Probability', font: { size: 11, color: '#1a9641' } },
      tickfont: { color: '#1a9641' },
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

  // Move the cursor trace
  if (chartEl && chartEl.data && chartEl.data.length >= 4) {
    Plotly.restyle('chart', { x: [[t, t]] }, [3]);
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

function buildEventList(mid, m, detail) {
  const el = document.getElementById('events-list');
  const header = document.getElementById('events-header');
  header.textContent = 'HDM Events — ' + mid + ' (TP=' + m.tp + ' FP=' + m.fp + ' FN=' + m.fn + ') — Click to jump & play';

  const pos = detail.samples.filter(s => s.label === 1);
  const fp = detail.samples.filter(s => s.label === 0 && s.pred === 1);

  let html = '';
  pos.forEach(s => {
    const isTP = s.pred === 1;
    const cls = isTP ? '' : ' fn';
    const tag = isTP
      ? '<span class="tag tag-tp">TP</span>'
      : '<span class="tag tag-fn">FN</span>';
    html += '<div class="event-row' + cls + '" onclick="jumpTo(' + s.time + ')">' +
      '<div class="ev-time">' + fmtTime(s.time) + '</div>' +
      '<div class="ev-text">"' + (s.text || '?') + '"</div>' +
      '<div class="ev-speaker">Speaker ' + s.speaker + '</div>' +
      tag +
      '<span class="tag tag-prob">P=' + s.prob_p.toFixed(3) + '</span>' +
      '</div>';
  });

  if (fp.length > 0) {
    html += '<div style="padding:8px 15px;font-size:12px;font-weight:600;color:#da3633;border-bottom:1px solid #f0f0f0">False Positives</div>';
    fp.forEach(s => {
      html += '<div class="event-row" onclick="jumpTo(' + s.time + ')" style="border-left:3px solid #da3633">' +
        '<div class="ev-time">' + fmtTime(s.time) + '</div>' +
        '<div class="ev-text">"' + (s.text || '') + '"</div>' +
        '<span class="tag tag-fp">FP</span>' +
        '<span class="tag tag-prob">P=' + s.prob_p.toFixed(3) + '</span>' +
        '</div>';
    });
  }

  el.innerHTML = html;
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
