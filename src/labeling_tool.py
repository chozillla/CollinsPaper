"""
HDM Labeling Tool — listen to potential hearing difficulty moments and label Yes/No.

For each of the 149 positive HDM candidates:
- Plays the 4s preceding context (what was happening before)
- Plays the 4s HDM segment (the moment itself)
- Plays a combined 8s clip (context + HDM together)
- You mark Yes (real HDM) or No (not HDM)

Labels are saved to data/hdm_labels.json after each decision.

Usage:
    python src/labeling_tool.py
    Then open http://localhost:8765 in your browser.
"""

import gc
import json
import io
import os
import numpy as np
import soundfile as sf
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import base64
from collections import defaultdict

AUDIO_DIR = Path("data/audio")
DATASET_DIR = Path("data/dataset")
LABELS_FILE = Path("data/hdm_labels.json")
CLIPS_DIR = Path("data/labeling_clips")
SAMPLE_RATE = 16000
SEGMENT_DURATION = 4.0
CONTEXT_DURATION = 4.0

# Global state
item_list = []  # list of {index, meta} — lightweight, no audio in memory
labels = {}  # str(idx) -> "yes" or "no"


def extract_clips():
    """Extract HDM + context clips as WAV files on disk (memory-efficient)."""
    global item_list

    CLIPS_DIR.mkdir(parents=True, exist_ok=True)

    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    positives = meta["positive"]
    print(f"Extracting audio for {len(positives)} HDM candidates...")

    # Group positives by meeting to load each meeting only once
    by_meeting = defaultdict(list)
    for i, pos in enumerate(positives):
        by_meeting[pos["meeting_id"]].append((i, pos))

    for mid, items in sorted(by_meeting.items()):
        # Check if all clips already exist
        all_exist = all(
            (CLIPS_DIR / f"{i}_hdm.wav").exists() and
            (CLIPS_DIR / f"{i}_ctx.wav").exists() and
            (CLIPS_DIR / f"{i}_combined.wav").exists()
            for i, _ in items
        )
        if all_exist:
            print(f"  {mid}: {len(items)} clips already extracted, skipping")
            for i, pos in items:
                item_list.append({"index": i, "meta": pos})
            continue

        audio_path = AUDIO_DIR / f"{mid}.Mix-Headset.wav"
        if not audio_path.exists():
            print(f"  SKIP: No audio for {mid}")
            continue

        audio, sr = sf.read(str(audio_path), dtype="float32")
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        for i, pos in items:
            sample_time = pos["sample_time"]
            hdm_end = int(sample_time * SAMPLE_RATE)
            hdm_start = int((sample_time - SEGMENT_DURATION) * SAMPLE_RATE)
            ctx_end = hdm_start
            ctx_start = int((sample_time - SEGMENT_DURATION - CONTEXT_DURATION) * SAMPLE_RATE)

            if ctx_start < 0 or hdm_end > len(audio):
                print(f"  SKIP: Out of bounds for {mid} at {sample_time}")
                continue

            hdm_audio = audio[hdm_start:hdm_end]
            ctx_audio = audio[ctx_start:ctx_end]
            combined_audio = audio[ctx_start:hdm_end]

            sf.write(str(CLIPS_DIR / f"{i}_hdm.wav"), hdm_audio, SAMPLE_RATE)
            sf.write(str(CLIPS_DIR / f"{i}_ctx.wav"), ctx_audio, SAMPLE_RATE)
            sf.write(str(CLIPS_DIR / f"{i}_combined.wav"), combined_audio, SAMPLE_RATE)

            item_list.append({"index": i, "meta": pos})

        del audio
        gc.collect()
        print(f"  {mid}: {len(items)} clips extracted")

    # Sort by index
    item_list.sort(key=lambda x: x["index"])
    print(f"\n{len(item_list)} clips ready for labeling.")


def wav_to_b64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def load_existing_labels():
    global labels
    if LABELS_FILE.exists():
        with open(LABELS_FILE) as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} existing labels from {LABELS_FILE}")


def save_labels():
    with open(LABELS_FILE, "w") as f:
        json.dump(labels, f, indent=2)


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>HDM Labeling Tool</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #1a1a2e; color: #eee; padding: 20px; }
  .container { max-width: 800px; margin: 0 auto; }
  h1 { text-align: center; margin-bottom: 10px; color: #e94560; }
  .progress-bar { background: #333; border-radius: 10px; height: 24px; margin: 15px 0; overflow: hidden; }
  .progress-fill { background: linear-gradient(90deg, #e94560, #0f3460); height: 100%; transition: width 0.3s; display: flex; align-items: center; justify-content: center; font-size: 12px; font-weight: bold; min-width: 30px; }
  .card { background: #16213e; border-radius: 12px; padding: 24px; margin: 20px 0; box-shadow: 0 4px 20px rgba(0,0,0,0.3); }
  .meta { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 20px; }
  .meta-item { background: #0f3460; padding: 8px 12px; border-radius: 6px; font-size: 14px; }
  .meta-label { color: #888; font-size: 11px; text-transform: uppercase; }
  .audio-section { margin: 16px 0; }
  .audio-label { font-size: 14px; color: #e94560; margin-bottom: 6px; font-weight: 600; }
  .audio-sublabel { font-size: 12px; color: #888; margin-bottom: 8px; }
  audio { width: 100%; height: 40px; }
  .text-display { background: #0f3460; padding: 16px; border-radius: 8px; margin: 16px 0; font-size: 18px; text-align: center; font-style: italic; min-height: 50px; display: flex; align-items: center; justify-content: center; }
  .buttons { display: flex; gap: 16px; margin-top: 24px; }
  .btn { flex: 1; padding: 18px; border: none; border-radius: 10px; font-size: 20px; font-weight: bold; cursor: pointer; transition: all 0.2s; }
  .btn:hover { transform: translateY(-2px); box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
  .btn:active { transform: translateY(0); }
  .btn-yes { background: #2ecc71; color: #fff; }
  .btn-yes:hover { background: #27ae60; }
  .btn-no { background: #e74c3c; color: #fff; }
  .btn-no:hover { background: #c0392b; }
  .nav { display: flex; gap: 10px; margin-top: 16px; justify-content: center; }
  .nav-btn { padding: 8px 20px; background: #0f3460; border: none; color: #eee; border-radius: 6px; cursor: pointer; font-size: 14px; }
  .nav-btn:hover { background: #1a4a8a; }
  .nav-btn:disabled { opacity: 0.3; cursor: default; }
  .status { text-align: center; margin: 10px 0; font-size: 13px; color: #888; }
  .labeled-yes { border: 2px solid #2ecc71; }
  .labeled-no { border: 2px solid #e74c3c; }
  .current-label { text-align: center; padding: 8px; border-radius: 6px; margin-top: 12px; font-weight: bold; }
  .keyboard-hint { text-align: center; color: #666; font-size: 12px; margin-top: 12px; }
  .loading { text-align: center; padding: 40px; color: #888; }
</style>
</head>
<body>
<div class="container">
  <h1>HDM Labeling Tool</h1>
  <div class="status" id="status">Loading...</div>
  <div class="progress-bar"><div class="progress-fill" id="progress">0%</div></div>

  <div class="card" id="card">
    <div class="meta" id="meta"></div>

    <div class="audio-section">
      <div class="audio-label">1. Context (preceding 4 seconds)</div>
      <div class="audio-sublabel">What was happening before the potential HDM</div>
      <audio id="audio-ctx" controls preload="auto"></audio>
    </div>

    <div class="audio-section">
      <div class="audio-label">2. HDM Segment (4 seconds)</div>
      <div class="audio-sublabel">The moment flagged as potential hearing difficulty</div>
      <audio id="audio-hdm" controls preload="auto"></audio>
    </div>

    <div class="audio-section">
      <div class="audio-label">3. Combined (8 seconds: context + HDM)</div>
      <div class="audio-sublabel">Full context flowing into the moment</div>
      <audio id="audio-combined" controls preload="auto"></audio>
    </div>

    <div class="text-display" id="text-display"></div>

    <div id="current-label"></div>

    <div class="buttons">
      <button class="btn btn-yes" onclick="labelItem('yes')">Yes — HDM</button>
      <button class="btn btn-no" onclick="labelItem('no')">No — Not HDM</button>
    </div>

    <div class="nav">
      <button class="nav-btn" id="prev-btn" onclick="navigate(-1)">&larr; Previous</button>
      <button class="nav-btn" onclick="jumpToUnlabeled()">Next Unlabeled</button>
      <button class="nav-btn" id="next-btn" onclick="navigate(1)">Next &rarr;</button>
    </div>

    <div class="keyboard-hint">Keyboard: Y = Yes, N = No, &larr; &rarr; = Navigate, Space = Play combined</div>
  </div>
</div>

<script>
let items = [];
let currentIdx = 0;
let labelMap = {};

async function init() {
  const resp = await fetch('/api/items');
  const data = await resp.json();
  items = data.items;
  labelMap = data.labels;
  const firstUnlabeled = items.findIndex(item => !(String(item.index) in labelMap));
  currentIdx = firstUnlabeled >= 0 ? firstUnlabeled : 0;
  await loadAndRender();
}

async function loadAndRender() {
  const item = items[currentIdx];
  const labeled = Object.keys(labelMap).length;
  const total = items.length;

  document.getElementById('status').textContent =
    `Item ${currentIdx + 1} of ${total} | ${labeled} labeled (${Math.round(labeled/total*100)}%)`;

  const pct = Math.round(labeled / total * 100);
  const progEl = document.getElementById('progress');
  progEl.style.width = Math.max(pct, 2) + '%';
  progEl.textContent = pct + '%';

  const meta = item.meta;
  document.getElementById('meta').innerHTML = `
    <div class="meta-item"><div class="meta-label">Meeting</div>${meta.meeting_id}</div>
    <div class="meta-item"><div class="meta-label">Speaker</div>${meta.speaker}</div>
    <div class="meta-item"><div class="meta-label">Time</div>${meta.sample_time.toFixed(1)}s</div>
    <div class="meta-item"><div class="meta-label">HDM Window</div>${meta.hdm_start?.toFixed(1) || '?'}s &mdash; ${meta.hdm_end?.toFixed(1) || '?'}s</div>
  `;

  const textEl = document.getElementById('text-display');
  textEl.textContent = meta.text ? '"' + meta.text + '"' : '(no transcript)';

  // Load audio for this item
  const audioResp = await fetch('/api/audio/' + item.index);
  const audioData = await audioResp.json();

  document.getElementById('audio-ctx').src = 'data:audio/wav;base64,' + audioData.ctx;
  document.getElementById('audio-hdm').src = 'data:audio/wav;base64,' + audioData.hdm;
  document.getElementById('audio-combined').src = 'data:audio/wav;base64,' + audioData.combined;

  const labelEl = document.getElementById('current-label');
  const card = document.getElementById('card');
  card.classList.remove('labeled-yes', 'labeled-no');
  const existingLabel = labelMap[String(item.index)];
  if (existingLabel) {
    card.classList.add(existingLabel === 'yes' ? 'labeled-yes' : 'labeled-no');
    labelEl.innerHTML = '<div class="current-label" style="background:' +
      (existingLabel === 'yes' ? '#2ecc71' : '#e74c3c') + '">Currently labeled: ' +
      existingLabel.toUpperCase() + '</div>';
  } else {
    labelEl.innerHTML = '';
  }

  document.getElementById('prev-btn').disabled = currentIdx === 0;
  document.getElementById('next-btn').disabled = currentIdx === items.length - 1;
}

async function labelItem(label) {
  const item = items[currentIdx];
  labelMap[String(item.index)] = label;

  await fetch('/api/label', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({index: item.index, label: label})
  });

  const nextUnlabeled = items.findIndex((item, idx) => idx > currentIdx && !(String(item.index) in labelMap));
  if (nextUnlabeled >= 0) {
    currentIdx = nextUnlabeled;
  } else if (currentIdx < items.length - 1) {
    currentIdx++;
  }
  await loadAndRender();
}

async function navigate(dir) {
  currentIdx = Math.max(0, Math.min(items.length - 1, currentIdx + dir));
  await loadAndRender();
}

function jumpToUnlabeled() {
  let next = items.findIndex((item, idx) => idx > currentIdx && !(String(item.index) in labelMap));
  if (next < 0) {
    next = items.findIndex(item => !(String(item.index) in labelMap));
  }
  if (next >= 0) {
    currentIdx = next;
    loadAndRender();
  }
}

document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === 'y' || e.key === 'Y') labelItem('yes');
  else if (e.key === 'n' || e.key === 'N') labelItem('no');
  else if (e.key === 'ArrowLeft') navigate(-1);
  else if (e.key === 'ArrowRight') navigate(1);
  else if (e.key === ' ') {
    e.preventDefault();
    document.getElementById('audio-combined').play();
  }
});

init();
</script>
</body>
</html>"""


class LabelingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode())

        elif parsed.path == "/api/items":
            # Send only metadata, no audio
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            response = {
                "items": item_list,
                "labels": labels,
            }
            self.wfile.write(json.dumps(response).encode())

        elif parsed.path.startswith("/api/audio/"):
            # Serve audio for a single item on demand
            try:
                idx = int(parsed.path.split("/")[-1])
                hdm_path = CLIPS_DIR / f"{idx}_hdm.wav"
                ctx_path = CLIPS_DIR / f"{idx}_ctx.wav"
                combined_path = CLIPS_DIR / f"{idx}_combined.wav"

                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(json.dumps({
                    "hdm": wav_to_b64(hdm_path),
                    "ctx": wav_to_b64(ctx_path),
                    "combined": wav_to_b64(combined_path),
                }).encode())
            except Exception as e:
                self.send_response(404)
                self.end_headers()
                self.wfile.write(str(e).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/api/label":
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length))
            idx = str(body["index"])
            label = body["label"]
            labels[idx] = label
            save_labels()

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"ok": True, "total_labeled": len(labels)}).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    print("HDM Labeling Tool")
    print("=" * 50)

    load_existing_labels()
    extract_clips()

    port = 8765
    server = HTTPServer(("0.0.0.0", port), LabelingHandler)
    print(f"\n{'=' * 50}")
    print(f"  Labeling UI ready at: http://localhost:{port}")
    print(f"{'=' * 50}")
    print(f"  {len(item_list)} HDM candidates to review")
    print(f"  {len(labels)} already labeled")
    print(f"\n  Keyboard shortcuts:")
    print(f"    Y = Yes (HDM)    N = No (not HDM)")
    print(f"    <- -> = Navigate   Space = Play combined")
    print(f"\n  Labels auto-save to {LABELS_FILE}")
    print(f"  Press Ctrl+C to stop.\n")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print(f"\nSaved {len(labels)} labels to {LABELS_FILE}")
        server.server_close()


if __name__ == "__main__":
    main()
