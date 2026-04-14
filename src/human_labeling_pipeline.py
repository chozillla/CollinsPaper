"""
Human HDM Labeling Pipeline — listen to full meeting audio and mark HDMs.

Unlike the AI-verified labeling tool (labeling_tool.py), this tool lets human
annotators listen to complete meeting audio and freely place HDM markers at
timestamps where they hear difficulty moments.

Two HDM types:
  Type A (Acoustic) — the listener physically misheard the audio
  Type B (Comprehension) — the listener heard but lacked language/comprehension

Multiple labelers supported. Labels saved to data/human_hdm_labels.json.

Usage:
    python src/human_labeling_pipeline.py
    python src/human_labeling_pipeline.py --labeler alice
    python src/human_labeling_pipeline.py --port 8770
    Then open http://localhost:8770
"""

import gc
import io
import json
import argparse
import base64
import numpy as np
import soundfile as sf
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from collections import defaultdict

ROOT = Path(__file__).parent.parent
AUDIO_DIR = ROOT / "data" / "audio"
RESULTS_DIR = ROOT / "results"
HDM_FILE = ROOT / "data" / "hdm_filtered.json"
LABELS_FILE = ROOT / "data" / "human_hdm_labels.json"
SAMPLE_RATE = 16000

# Global state
meetings = {}       # mid -> {duration, n_ground_truth}
labels_db = {}      # loaded from LABELS_FILE
current_labeler = "default"


def discover_meetings():
    """Find all meetings with audio files."""
    global meetings

    # Load ground truth HDMs for reference counts
    gt_counts = defaultdict(int)
    if HDM_FILE.exists():
        with open(HDM_FILE) as f:
            for hdm in json.load(f):
                gt_counts[hdm["meeting_id"]] += 1

    for wav in sorted(AUDIO_DIR.glob("*.Mix-Headset.wav")):
        mid = wav.name.replace(".Mix-Headset.wav", "")
        info = sf.info(str(wav))
        meetings[mid] = {
            "duration": info.duration,
            "n_ground_truth": gt_counts.get(mid, 0),
        }

    print(f"Found {len(meetings)} meetings with audio.")


def load_labels():
    global labels_db
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            labels_db = json.load(f)
    else:
        labels_db = {}


def save_labels():
    with open(LABELS_FILE, "w") as f:
        json.dump(labels_db, f, indent=2)


def get_labeler_data(labeler, meeting_id):
    """Get labels for a specific labeler and meeting."""
    return labels_db.get(labeler, {}).get(meeting_id, [])


def set_labeler_data(labeler, meeting_id, hdm_list):
    """Save labels for a specific labeler and meeting."""
    if labeler not in labels_db:
        labels_db[labeler] = {}
    labels_db[labeler][meeting_id] = hdm_list
    save_labels()


def get_meeting_audio_b64(meeting_id):
    """Load full meeting audio as base64 WAV."""
    wav_path = AUDIO_DIR / f"{meeting_id}.Mix-Headset.wav"
    if not wav_path.exists():
        return None
    with open(wav_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_meeting_waveform(meeting_id, target_points=4000):
    """Get downsampled waveform envelope for display."""
    wav_path = AUDIO_DIR / f"{meeting_id}.Mix-Headset.wav"
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    chunk = max(1, len(audio) // target_points)
    n_chunks = len(audio) // chunk
    audio = audio[:n_chunks * chunk].reshape(n_chunks, chunk)
    envelope = np.max(np.abs(audio), axis=1)
    envelope = (envelope / (envelope.max() + 1e-9)).tolist()

    duration = len(audio) * chunk / sr
    times = np.linspace(0, duration, n_chunks).tolist()

    return {"times": times, "envelope": envelope}


def get_sliding_window_data(meeting_id):
    """Load AI probability signal if available."""
    sw_path = RESULTS_DIR / "sliding_window_10shot" / f"{meeting_id}.json"
    if not sw_path.exists():
        sw_path = RESULTS_DIR / "sliding_window" / f"{meeting_id}.json"
    if not sw_path.exists():
        return None
    with open(sw_path) as f:
        data = json.load(f)
    windows = data.get("windows", [])
    return {
        "times": [w["time"] for w in windows],
        "probs": [w["prob_p"] for w in windows],
    }


def get_ground_truth_hdms(meeting_id):
    """Load ground truth HDM annotations for this meeting."""
    if not HDM_FILE.exists():
        return []
    with open(HDM_FILE) as f:
        all_hdms = json.load(f)
    return [
        {
            "start": h["start_time"],
            "end": h["end_time"],
            "text": h.get("text", ""),
            "speaker": h.get("speaker", ""),
        }
        for h in all_hdms if h["meeting_id"] == meeting_id
    ]


def get_clip_b64(meeting_id, center_time, radius=6.0):
    """Extract a clip around a timestamp and return as base64 WAV."""
    wav_path = AUDIO_DIR / f"{meeting_id}.Mix-Headset.wav"
    audio, sr = sf.read(str(wav_path), dtype="float32")
    if len(audio.shape) > 1:
        audio = audio.mean(axis=1)

    start = max(0, int((center_time - radius) * sr))
    end = min(len(audio), int((center_time + radius) * sr))
    clip = audio[start:end]

    buf = io.BytesIO()
    sf.write(buf, clip, sr, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Human HDM Labeling Pipeline</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #fff; color: #222; }

  .header { padding: 24px 32px; border-bottom: 1px solid #e0e0e0; background: #fafbfc; }
  .header h1 { font-size: 22px; font-weight: 700; color: #333; }
  .header p { color: #666; font-size: 14px; margin-top: 4px; }
  .labeler-badge { display: inline-block; background: #e8f4fd; color: #1976d2; padding: 2px 10px; border-radius: 12px; font-size: 13px; margin-left: 12px; font-weight: 600; }

  .layout { display: flex; height: calc(100vh - 85px); }

  /* Meeting list sidebar */
  .sidebar { width: 280px; border-right: 1px solid #e0e0e0; overflow-y: auto; background: #fafbfc; flex-shrink: 0; }
  .sidebar-header { padding: 12px 16px; font-size: 13px; font-weight: 600; color: #888; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #e0e0e0; }
  .meeting-item { padding: 10px 16px; cursor: pointer; border-bottom: 1px solid #f0f0f0; transition: background 0.15s; }
  .meeting-item:hover { background: #e8f4fd; }
  .meeting-item.active { background: #d4ecfc; border-left: 3px solid #1976d2; }
  .meeting-id { font-weight: 600; font-size: 14px; }
  .meeting-meta { font-size: 12px; color: #888; margin-top: 2px; }
  .meeting-labels-count { font-size: 11px; color: #1976d2; font-weight: 600; }

  /* Main panel */
  .main { flex: 1; overflow-y: auto; padding: 24px 32px; }

  .instructions { background: #f8f9fa; border: 1px solid #e0e0e0; border-radius: 8px; padding: 16px 20px; margin-bottom: 20px; }
  .instructions h3 { font-size: 14px; margin-bottom: 8px; color: #333; }
  .instructions p { font-size: 13px; color: #555; line-height: 1.5; }
  .type-legend { display: flex; gap: 24px; margin-top: 10px; }
  .type-a { color: #d32f2f; font-weight: 600; }
  .type-b { color: #7b1fa2; font-weight: 600; }

  /* Waveform canvas */
  .waveform-container { position: relative; background: #fff; border: 1px solid #ccc; border-radius: 4px; margin-bottom: 16px; }
  .waveform-container canvas { display: block; width: 100%; cursor: crosshair; }

  /* Audio player */
  .audio-controls { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding: 12px 16px; background: #f8f9fa; border-radius: 8px; border: 1px solid #e0e0e0; }
  .audio-controls button { padding: 6px 16px; border: 1px solid #ccc; background: #fff; border-radius: 4px; cursor: pointer; font-size: 13px; }
  .audio-controls button:hover { background: #f0f0f0; }
  .time-display { font-family: monospace; font-size: 14px; color: #333; min-width: 140px; }
  .speed-controls { margin-left: auto; display: flex; gap: 4px; }
  .speed-btn { padding: 4px 8px !important; font-size: 12px !important; }
  .speed-btn.active { background: #1976d2 !important; color: #fff !important; border-color: #1976d2 !important; }

  /* HDM label list */
  .labels-section { margin-top: 20px; }
  .labels-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 12px; }
  .labels-header h3 { font-size: 16px; color: #333; }
  .label-count { font-size: 13px; color: #888; }

  .hdm-label-card { display: flex; align-items: center; gap: 12px; padding: 10px 14px; background: #fff; border: 1px solid #e0e0e0; border-radius: 6px; margin-bottom: 8px; transition: background 0.15s; }
  .hdm-label-card:hover { background: #f8f9fa; }
  .hdm-type-badge { padding: 3px 10px; border-radius: 10px; font-size: 12px; font-weight: 700; color: #fff; min-width: 55px; text-align: center; }
  .hdm-type-badge.type-a { background: #d32f2f; }
  .hdm-type-badge.type-b { background: #7b1fa2; }
  .hdm-time { font-family: monospace; font-size: 14px; color: #333; min-width: 60px; }
  .hdm-note { flex: 1; font-size: 13px; color: #555; }
  .hdm-actions { display: flex; gap: 6px; }
  .hdm-actions button { padding: 4px 10px; border: 1px solid #ccc; background: #fff; border-radius: 4px; cursor: pointer; font-size: 12px; }
  .hdm-actions button:hover { background: #f0f0f0; }
  .hdm-actions button.delete { color: #d32f2f; border-color: #d32f2f; }
  .hdm-actions button.delete:hover { background: #fde8e8; }

  /* Label placement overlay */
  .label-dialog { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(0,0,0,0.3); z-index: 100; align-items: center; justify-content: center; }
  .label-dialog.open { display: flex; }
  .label-dialog-box { background: #fff; border-radius: 12px; padding: 24px; width: 420px; box-shadow: 0 8px 30px rgba(0,0,0,0.2); }
  .label-dialog-box h3 { margin-bottom: 4px; }
  .label-dialog-box .time-info { font-family: monospace; color: #666; margin-bottom: 16px; font-size: 14px; }
  .label-dialog-box .clip-player { margin-bottom: 16px; }
  .label-dialog-box audio { width: 100%; height: 36px; }
  .type-buttons { display: flex; gap: 12px; margin-bottom: 16px; }
  .type-btn { flex: 1; padding: 14px; border: 2px solid #ccc; background: #fff; border-radius: 8px; cursor: pointer; text-align: center; transition: all 0.15s; }
  .type-btn:hover { border-color: #999; }
  .type-btn.selected-a { border-color: #d32f2f; background: #fde8e8; }
  .type-btn.selected-b { border-color: #7b1fa2; background: #f3e5f5; }
  .type-btn .type-letter { font-size: 24px; font-weight: 700; }
  .type-btn .type-desc { font-size: 12px; color: #666; margin-top: 4px; }
  .note-input { width: 100%; padding: 8px 12px; border: 1px solid #ccc; border-radius: 6px; font-size: 14px; margin-bottom: 16px; }
  .dialog-actions { display: flex; gap: 10px; justify-content: flex-end; }
  .dialog-actions button { padding: 8px 20px; border-radius: 6px; cursor: pointer; font-size: 14px; border: 1px solid #ccc; background: #fff; }
  .dialog-actions .save-btn { background: #1976d2; color: #fff; border-color: #1976d2; }
  .dialog-actions .save-btn:disabled { opacity: 0.4; cursor: default; }

  .empty-state { text-align: center; padding: 60px 20px; color: #999; }
  .empty-state h3 { font-size: 18px; margin-bottom: 8px; color: #666; }

  /* Ground truth markers */
  .gt-legend { display: flex; gap: 16px; margin-bottom: 8px; font-size: 12px; color: #888; }
  .gt-legend span { display: flex; align-items: center; gap: 4px; }
  .gt-swatch { width: 14px; height: 10px; border-radius: 2px; display: inline-block; }
</style>
</head>
<body>

<div class="header">
  <h1>Human HDM Labeling Pipeline <span class="labeler-badge" id="labeler-badge"></span></h1>
  <p>Listen to meeting audio. Click the waveform to mark Hearing Difficulty Moments.</p>
</div>

<div class="layout">
  <div class="sidebar">
    <div class="sidebar-header">Meetings</div>
    <div id="meeting-list"></div>
  </div>

  <div class="main" id="main-panel">
    <div class="empty-state" id="empty-state">
      <h3>Select a meeting</h3>
      <p>Pick a meeting from the sidebar to start labeling.</p>
    </div>

    <div id="meeting-view" style="display:none;">
      <div class="instructions">
        <h3>How to label</h3>
        <p>
          Play the audio and listen for moments where a participant struggles to understand.
          <strong>Click the waveform</strong> at the timestamp where you hear it, then choose the type.
        </p>
        <div class="type-legend">
          <span class="type-a">Type A &mdash; Acoustic: misheard the audio (unclear speech, noise, overlap)</span>
        </div>
        <div class="type-legend" style="margin-top:4px;">
          <span class="type-b">Type B &mdash; Comprehension: heard the words but didn't understand (language, jargon, context)</span>
        </div>
      </div>

      <div class="gt-legend">
        <span><span class="gt-swatch" style="background:rgba(211,47,47,0.15);border:1px solid rgba(211,47,47,0.4);"></span> Ground truth HDM (AI-identified)</span>
        <span><span class="gt-swatch" style="background:rgba(25,118,210,0.3);border:1px solid rgba(25,118,210,0.5);"></span> AI P(HDM) signal</span>
        <span><span class="gt-swatch" style="background:#d32f2f;"></span> Your Type A labels</span>
        <span><span class="gt-swatch" style="background:#7b1fa2;"></span> Your Type B labels</span>
      </div>

      <div class="waveform-container">
        <canvas id="waveform-canvas" height="280"></canvas>
      </div>

      <div class="audio-controls">
        <button onclick="togglePlay()" id="play-btn">&#9654; Play</button>
        <button onclick="skipBy(-5)">-5s</button>
        <button onclick="skipBy(5)">+5s</button>
        <span class="time-display" id="time-display">0:00.0 / 0:00.0</span>
        <div class="speed-controls">
          <button class="speed-btn" onclick="setSpeed(0.5)">0.5x</button>
          <button class="speed-btn active" onclick="setSpeed(1)">1x</button>
          <button class="speed-btn" onclick="setSpeed(1.5)">1.5x</button>
          <button class="speed-btn" onclick="setSpeed(2)">2x</button>
        </div>
      </div>

      <div class="labels-section">
        <div class="labels-header">
          <h3>Your HDM Labels</h3>
          <span class="label-count" id="label-count"></span>
        </div>
        <div id="labels-list"></div>
      </div>
    </div>
  </div>
</div>

<!-- Label placement dialog -->
<div class="label-dialog" id="label-dialog">
  <div class="label-dialog-box">
    <h3>Mark HDM</h3>
    <div class="time-info" id="dialog-time"></div>
    <div class="clip-player">
      <audio id="dialog-audio" controls preload="auto" style="width:100%;height:36px;"></audio>
    </div>
    <div class="type-buttons">
      <div class="type-btn" id="btn-type-a" onclick="selectType('A')">
        <div class="type-letter" style="color:#d32f2f;">A</div>
        <div class="type-desc">Acoustic — misheard the audio</div>
      </div>
      <div class="type-btn" id="btn-type-b" onclick="selectType('B')">
        <div class="type-letter" style="color:#7b1fa2;">B</div>
        <div class="type-desc">Comprehension — heard but didn't understand</div>
      </div>
    </div>
    <input class="note-input" id="note-input" type="text" placeholder="Optional note (e.g., overlapping speakers, strong accent)...">
    <div class="dialog-actions">
      <button onclick="closeDialog()">Cancel</button>
      <button class="save-btn" id="save-label-btn" onclick="saveLabel()" disabled>Save Label</button>
    </div>
  </div>
</div>

<script>
const LABELER = '__LABELER__';
let meetingList = [];
let currentMeeting = null;
let currentLabels = [];
let groundTruth = [];
let aiSignal = null;
let waveformData = null;
let audioEl = null;
let audioCtx = null;
let pendingTime = null;
let selectedType = null;

document.getElementById('labeler-badge').textContent = LABELER;

// ---- Init ----
async function init() {
  const resp = await fetch('/api/meetings');
  meetingList = await resp.json();
  renderMeetingList();
}

function renderMeetingList() {
  const el = document.getElementById('meeting-list');
  el.innerHTML = meetingList.map(m => `
    <div class="meeting-item" id="mi-${m.id}" onclick="selectMeeting('${m.id}')">
      <div class="meeting-id">${m.id}</div>
      <div class="meeting-meta">${formatTime(m.duration)} &middot; ${m.n_ground_truth} GT HDMs</div>
      <div class="meeting-labels-count">${m.n_human_labels} human labels</div>
    </div>
  `).join('');
}

async function selectMeeting(mid) {
  document.querySelectorAll('.meeting-item').forEach(el => el.classList.remove('active'));
  document.getElementById('mi-' + mid).classList.add('active');
  document.getElementById('empty-state').style.display = 'none';
  document.getElementById('meeting-view').style.display = 'block';

  currentMeeting = mid;

  // Load meeting data in parallel
  const [labelsResp, waveResp, gtResp, signalResp] = await Promise.all([
    fetch('/api/labels/' + mid),
    fetch('/api/waveform/' + mid),
    fetch('/api/ground_truth/' + mid),
    fetch('/api/signal/' + mid),
  ]);

  currentLabels = await labelsResp.json();
  waveformData = await waveResp.json();
  groundTruth = await gtResp.json();
  const signalData = await signalResp.json();
  aiSignal = signalData.times ? signalData : null;

  // Load audio
  const audioResp = await fetch('/api/audio/' + mid);
  const audioBlob = await audioResp.blob();
  const audioUrl = URL.createObjectURL(audioBlob);

  if (audioEl) { audioEl.pause(); URL.revokeObjectURL(audioEl.src); }
  audioEl = new Audio(audioUrl);
  audioEl.addEventListener('timeupdate', onTimeUpdate);

  drawWaveform();
  renderLabels();
}

// ---- Waveform drawing ----
function drawWaveform() {
  const canvas = document.getElementById('waveform-canvas');
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width;
  canvas.height = 280;
  const ctx = canvas.getContext('2d');
  const W = canvas.width, H = canvas.height;
  const dur = waveformData.times[waveformData.times.length - 1];

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#fff';
  ctx.fillRect(0, 0, W, H);

  // Ground truth HDM bands
  for (const h of groundTruth) {
    const x1 = (h.start / dur) * W;
    const x2 = (h.end / dur) * W;
    ctx.fillStyle = 'rgba(211, 47, 47, 0.12)';
    ctx.fillRect(x1, 0, Math.max(x2 - x1, 3), H);
    ctx.strokeStyle = 'rgba(211, 47, 47, 0.3)';
    ctx.lineWidth = 1;
    ctx.strokeRect(x1, 0, Math.max(x2 - x1, 3), H);
  }

  // Waveform
  const mid = H / 2;
  ctx.beginPath();
  ctx.strokeStyle = '#666';
  ctx.lineWidth = 0.8;
  const env = waveformData.envelope;
  for (let i = 0; i < env.length; i++) {
    const x = (i / env.length) * W;
    const amp = env[i] * (H * 0.4);
    if (i === 0) { ctx.moveTo(x, mid - amp); } else { ctx.lineTo(x, mid - amp); }
  }
  for (let i = env.length - 1; i >= 0; i--) {
    const x = (i / env.length) * W;
    const amp = env[i] * (H * 0.4);
    ctx.lineTo(x, mid + amp);
  }
  ctx.closePath();
  ctx.fillStyle = 'rgba(100, 100, 100, 0.15)';
  ctx.fill();
  ctx.stroke();

  // AI probability signal
  if (aiSignal) {
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(25, 118, 210, 0.5)';
    ctx.lineWidth = 1.5;
    for (let i = 0; i < aiSignal.times.length; i++) {
      const x = (aiSignal.times[i] / dur) * W;
      const y = H - aiSignal.probs[i] * H;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
  }

  // Human labels — markers
  for (const lbl of currentLabels) {
    const x = (lbl.time / dur) * W;
    ctx.beginPath();
    ctx.strokeStyle = lbl.type === 'A' ? '#d32f2f' : '#7b1fa2';
    ctx.lineWidth = 2.5;
    ctx.moveTo(x, 0);
    ctx.lineTo(x, H);
    ctx.stroke();
    // Triangle marker at top
    ctx.fillStyle = lbl.type === 'A' ? '#d32f2f' : '#7b1fa2';
    ctx.beginPath();
    ctx.moveTo(x - 6, 0);
    ctx.lineTo(x + 6, 0);
    ctx.lineTo(x, 12);
    ctx.closePath();
    ctx.fill();
    // Type letter
    ctx.fillStyle = '#fff';
    ctx.font = 'bold 8px sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(lbl.type, x, 9);
  }

  // Playback cursor
  if (audioEl && audioEl.duration) {
    const cx = (audioEl.currentTime / audioEl.duration) * W;
    ctx.beginPath();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1.5;
    ctx.setLineDash([4, 3]);
    ctx.moveTo(cx, 0);
    ctx.lineTo(cx, H);
    ctx.stroke();
    ctx.setLineDash([]);
  }

  // Time axis labels
  ctx.fillStyle = '#999';
  ctx.font = '11px monospace';
  ctx.textAlign = 'center';
  const step = dur > 1800 ? 300 : dur > 600 ? 120 : 60;
  for (let t = 0; t <= dur; t += step) {
    const x = (t / dur) * W;
    ctx.fillText(formatTime(t), x, H - 4);
  }
}

function onTimeUpdate() {
  drawWaveform();
  if (audioEl) {
    const cur = audioEl.currentTime;
    const dur = audioEl.duration || 0;
    document.getElementById('time-display').textContent =
      formatTime(cur) + ' / ' + formatTime(dur);
  }
}

// ---- Waveform click → place label ----
document.getElementById('waveform-canvas').addEventListener('click', function(e) {
  if (!audioEl || !waveformData) return;
  const rect = this.getBoundingClientRect();
  const x = e.clientX - rect.left;
  const frac = x / rect.width;
  const dur = waveformData.times[waveformData.times.length - 1];
  const clickTime = frac * dur;

  // Seek audio to that point
  audioEl.currentTime = clickTime;
  drawWaveform();

  // Open label dialog
  openDialog(clickTime);
});

// ---- Dialog ----
function openDialog(time) {
  pendingTime = time;
  selectedType = null;
  document.getElementById('dialog-time').textContent = 'Time: ' + formatTime(time);
  document.getElementById('note-input').value = '';
  document.getElementById('btn-type-a').className = 'type-btn';
  document.getElementById('btn-type-b').className = 'type-btn';
  document.getElementById('save-label-btn').disabled = true;
  document.getElementById('label-dialog').classList.add('open');

  // Load clip audio
  fetch('/api/clip/' + currentMeeting + '/' + time.toFixed(2))
    .then(r => r.json())
    .then(data => {
      document.getElementById('dialog-audio').src = 'data:audio/wav;base64,' + data.clip;
    });
}

function closeDialog() {
  document.getElementById('label-dialog').classList.remove('open');
  pendingTime = null;
  selectedType = null;
}

function selectType(type) {
  selectedType = type;
  document.getElementById('btn-type-a').className = 'type-btn' + (type === 'A' ? ' selected-a' : '');
  document.getElementById('btn-type-b').className = 'type-btn' + (type === 'B' ? ' selected-b' : '');
  document.getElementById('save-label-btn').disabled = false;
}

async function saveLabel() {
  if (!selectedType || pendingTime === null) return;
  const label = {
    time: Math.round(pendingTime * 100) / 100,
    type: selectedType,
    note: document.getElementById('note-input').value.trim(),
  };

  const resp = await fetch('/api/labels/' + currentMeeting, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(label),
  });
  currentLabels = await resp.json();

  closeDialog();
  drawWaveform();
  renderLabels();
  updateMeetingListCount();
}

async function deleteLabel(idx) {
  const resp = await fetch('/api/labels/' + currentMeeting + '/delete', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({index: idx}),
  });
  currentLabels = await resp.json();
  drawWaveform();
  renderLabels();
  updateMeetingListCount();
}

function seekToLabel(time) {
  if (audioEl) {
    audioEl.currentTime = time;
    audioEl.play();
    document.getElementById('play-btn').innerHTML = '&#9646;&#9646; Pause';
    drawWaveform();
  }
}

// ---- Label list rendering ----
function renderLabels() {
  const sorted = [...currentLabels].sort((a, b) => a.time - b.time);
  document.getElementById('label-count').textContent =
    sorted.length + ' label' + (sorted.length !== 1 ? 's' : '');

  if (sorted.length === 0) {
    document.getElementById('labels-list').innerHTML =
      '<p style="color:#999;font-size:13px;padding:12px 0;">No labels yet. Click the waveform to add one.</p>';
    return;
  }

  document.getElementById('labels-list').innerHTML = sorted.map((lbl, i) => {
    const realIdx = currentLabels.indexOf(lbl);
    return `
    <div class="hdm-label-card">
      <span class="hdm-type-badge type-${lbl.type.toLowerCase()}">${lbl.type}</span>
      <span class="hdm-time">${formatTime(lbl.time)}</span>
      <span class="hdm-note">${lbl.note || '<span style="color:#ccc;">no note</span>'}</span>
      <div class="hdm-actions">
        <button onclick="seekToLabel(${lbl.time})">&#9654; Play</button>
        <button class="delete" onclick="deleteLabel(${realIdx})">Delete</button>
      </div>
    </div>`;
  }).join('');
}

function updateMeetingListCount() {
  const el = document.getElementById('mi-' + currentMeeting);
  if (el) {
    el.querySelector('.meeting-labels-count').textContent = currentLabels.length + ' human labels';
  }
}

// ---- Audio controls ----
function togglePlay() {
  if (!audioEl) return;
  if (audioEl.paused) {
    audioEl.play();
    document.getElementById('play-btn').innerHTML = '&#9646;&#9646; Pause';
  } else {
    audioEl.pause();
    document.getElementById('play-btn').innerHTML = '&#9654; Play';
  }
}

function skipBy(sec) {
  if (!audioEl) return;
  audioEl.currentTime = Math.max(0, Math.min(audioEl.duration, audioEl.currentTime + sec));
  drawWaveform();
}

function setSpeed(rate) {
  if (!audioEl) return;
  audioEl.playbackRate = rate;
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.classList.toggle('active', parseFloat(btn.textContent) === rate);
  });
}

// ---- Keyboard shortcuts ----
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (document.getElementById('label-dialog').classList.contains('open')) {
    if (e.key === 'a' || e.key === 'A') selectType('A');
    else if (e.key === 'b' || e.key === 'B') selectType('B');
    else if (e.key === 'Escape') closeDialog();
    else if (e.key === 'Enter' && selectedType) saveLabel();
    return;
  }
  if (e.key === ' ') { e.preventDefault(); togglePlay(); }
  else if (e.key === 'ArrowLeft') skipBy(-5);
  else if (e.key === 'ArrowRight') skipBy(5);
});

// ---- Helpers ----
function formatTime(s) {
  const m = Math.floor(s / 60);
  const sec = (s % 60).toFixed(1);
  return m + ':' + (sec < 10 ? '0' : '') + sec;
}

// Resize handling
window.addEventListener('resize', () => { if (waveformData) drawWaveform(); });

init();
</script>
</body>
</html>"""


class PipelineHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._html(HTML_TEMPLATE.replace("__LABELER__", current_labeler))

        elif path == "/api/meetings":
            result = []
            for mid, info in sorted(meetings.items()):
                n_labels = len(get_labeler_data(current_labeler, mid))
                result.append({
                    "id": mid,
                    "duration": round(info["duration"], 1),
                    "n_ground_truth": info["n_ground_truth"],
                    "n_human_labels": n_labels,
                })
            self._json(result)

        elif path.startswith("/api/labels/"):
            mid = path.split("/")[-1]
            self._json(get_labeler_data(current_labeler, mid))

        elif path.startswith("/api/waveform/"):
            mid = path.split("/")[-1]
            self._json(get_meeting_waveform(mid))

        elif path.startswith("/api/ground_truth/"):
            mid = path.split("/")[-1]
            self._json(get_ground_truth_hdms(mid))

        elif path.startswith("/api/signal/"):
            mid = path.split("/")[-1]
            data = get_sliding_window_data(mid)
            self._json(data if data else {})

        elif path.startswith("/api/audio/"):
            mid = path.split("/")[-1]
            wav_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
            if wav_path.exists():
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.end_headers()
                with open(wav_path, "rb") as f:
                    while True:
                        chunk = f.read(65536)
                        if not chunk:
                            break
                        self.wfile.write(chunk)
            else:
                self.send_response(404)
                self.end_headers()

        elif path.startswith("/api/clip/"):
            parts = path.split("/")
            mid = parts[3]
            t = float(parts[4])
            clip = get_clip_b64(mid, t)
            self._json({"clip": clip})

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path.endswith("/delete"):
            mid = path.split("/")[-2]
            idx = body["index"]
            hdms = get_labeler_data(current_labeler, mid)
            if 0 <= idx < len(hdms):
                hdms.pop(idx)
            set_labeler_data(current_labeler, mid, hdms)
            self._json(hdms)

        elif path.startswith("/api/labels/"):
            mid = path.split("/")[-1]
            hdms = get_labeler_data(current_labeler, mid)
            hdms.append(body)
            set_labeler_data(current_labeler, mid, hdms)
            self._json(hdms)

        else:
            self.send_response(404)
            self.end_headers()

    def _json(self, data):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())

    def _html(self, content):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(content.encode())

    def log_message(self, fmt, *args):
        pass


def main():
    global current_labeler

    parser = argparse.ArgumentParser(description="Human HDM Labeling Pipeline")
    parser.add_argument("--labeler", default="default", help="Labeler name/ID")
    parser.add_argument("--port", type=int, default=8770, help="Server port")
    args = parser.parse_args()

    current_labeler = args.labeler

    print("Human HDM Labeling Pipeline")
    print("=" * 50)
    print(f"  Labeler: {current_labeler}")

    load_labels()
    discover_meetings()

    n_existing = sum(
        len(v) for v in labels_db.get(current_labeler, {}).values()
    )

    server = HTTPServer(("0.0.0.0", args.port), PipelineHandler)
    print(f"\n  Ready at: http://localhost:{args.port}")
    print(f"  {len(meetings)} meetings available")
    print(f"  {n_existing} existing labels for '{current_labeler}'")
    print(f"\n  Keyboard shortcuts:")
    print(f"    Space = Play/Pause   Arrow keys = Skip +/-5s")
    print(f"    In dialog: A = Type A, B = Type B, Enter = Save, Esc = Cancel")
    print(f"\n  Labels auto-save to {LABELS_FILE}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nDone. Labels saved to {LABELS_FILE}")
        server.server_close()


if __name__ == "__main__":
    main()
