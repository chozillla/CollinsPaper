"""
Download AMI corpus headset mix audio files for meetings that contain HDMs.
Uses the AMI corpus download server.
"""

import json
import subprocess
import os
from pathlib import Path
from collections import defaultdict


AUDIO_DIR = Path("data/audio")
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# AMI audio download base URL for headset mix
# Format: https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus/{meeting_id}/audio/{meeting_id}.Mix-Headset.wav
BASE_URL = "https://groups.inf.ed.ac.uk/ami/AMICorpusMirror/amicorpus"


def get_required_meetings():
    """Get list of meetings that contain HDMs."""
    with open("data/hdm_filtered.json") as f:
        hdms = json.load(f)

    meetings = defaultdict(int)
    for h in hdms:
        meetings[h["meeting_id"]] += 1

    return dict(meetings)


def download_meeting_audio(meeting_id):
    """Download the headset mix WAV for a meeting."""
    filename = f"{meeting_id}.Mix-Headset.wav"
    output_path = AUDIO_DIR / filename

    if output_path.exists():
        print(f"  Already downloaded: {filename}")
        return True

    url = f"{BASE_URL}/{meeting_id}/audio/{filename}"
    print(f"  Downloading: {filename}...")

    try:
        result = subprocess.run(
            ["curl", "-L", "-f", "-o", str(output_path), url],
            capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            print(f"    FAILED: {result.stderr[:200]}")
            output_path.unlink(missing_ok=True)
            return False
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"    OK ({size_mb:.1f} MB)")
        return True
    except subprocess.TimeoutExpired:
        print(f"    TIMEOUT")
        output_path.unlink(missing_ok=True)
        return False


def main():
    meetings = get_required_meetings()
    print(f"Need audio for {len(meetings)} meetings")
    print(f"HDM counts: min={min(meetings.values())}, max={max(meetings.values())}, "
          f"total={sum(meetings.values())}")

    # Sort by HDM count (most HDMs first)
    sorted_meetings = sorted(meetings.items(), key=lambda x: -x[1])

    success = 0
    failed = 0

    for meeting_id, hdm_count in sorted_meetings:
        print(f"\n[{meeting_id}] ({hdm_count} HDMs)")
        if download_meeting_audio(meeting_id):
            success += 1
        else:
            failed += 1

    print(f"\n=== Download complete ===")
    print(f"Success: {success}/{len(meetings)}")
    print(f"Failed: {failed}/{len(meetings)}")


if __name__ == "__main__":
    main()
