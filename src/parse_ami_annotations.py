"""
Parse AMI corpus NXT XML annotations to extract Hearing Difficulty Moments.

Maps dialogue act annotations to word-level timestamps to identify
"Comment-About-Understanding" (ami_da_12) and "Elicit-Comment-Understanding" (ami_da_13)
utterances — analogous to "signal-non-understanding" in SWDA/MRDA used by Collins et al.
"""

import xml.etree.ElementTree as ET
import os
import re
import json
import csv
from pathlib import Path
from collections import defaultdict


ANNOTATION_DIR = Path("data/annotations")
DA_DIR = ANNOTATION_DIR / "dialogueActs"
WORDS_DIR = ANNOTATION_DIR / "words"

# Target dialogue act IDs (from da-types.xml)
# ami_da_12 = "und" = Comment-About-Understanding
# ami_da_13 = "el.und" = Elicit-Comment-Understanding
TARGET_DA_IDS = {"ami_da_12", "ami_da_13"}

NS = {"nite": "http://nite.sourceforge.net/"}


def parse_words_file(words_path):
    """Parse a words XML file and return dict of word_id -> {text, starttime, endtime}."""
    tree = ET.parse(words_path)
    root = tree.getroot()
    words = {}
    for elem in root:
        tag = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        word_id = elem.get(f"{{{NS['nite']}}}id") or elem.get("nite:id")
        if word_id is None:
            for k, v in elem.attrib.items():
                if k.endswith("}id") or k == "nite:id":
                    word_id = v
                    break
        if word_id is None:
            continue

        starttime = elem.get("starttime")
        endtime = elem.get("endtime")
        text = elem.text if elem.text else ""

        # Handle vocalsound, pause, etc.
        if tag in ("vocalsound", "nonvocalsound", "pause", "comment", "gap"):
            text = f"[{tag}:{elem.get('type', '')}]"

        words[word_id] = {
            "text": text.strip(),
            "starttime": float(starttime) if starttime else None,
            "endtime": float(endtime) if endtime else None,
        }
    return words


def parse_child_href(href):
    """Parse nite:child href to extract file and word ID range.

    Examples:
        'ES2002a.A.words.xml#id(ES2002a.A.words0)..id(ES2002a.A.words12)'
        'ES2002a.A.words.xml#id(ES2002a.A.words42)'
    """
    match = re.match(r'(.+\.words\.xml)#id\((.+?)\)(?:\.\.id\((.+?)\))?', href)
    if not match:
        return None, None, None
    words_file = match.group(1)
    start_id = match.group(2)
    end_id = match.group(3)  # None if single word
    return words_file, start_id, end_id


def get_da_type_id(pointer_elem):
    """Extract dialogue act type ID from a nite:pointer element."""
    href = pointer_elem.get("href", "")
    match = re.search(r'id\((.+?)\)', href)
    return match.group(1) if match else None


def parse_dialogue_acts(da_path, words_cache):
    """Parse a dialogue act XML file and extract HDM utterances with timestamps."""
    tree = ET.parse(da_path)
    root = tree.getroot()

    # Derive meeting and speaker from filename
    # e.g., ES2002a.A.dialog-act.xml -> meeting=ES2002a, speaker=A
    basename = os.path.basename(da_path)
    parts = basename.replace(".dialog-act.xml", "").split(".")
    meeting_id = parts[0]
    speaker = parts[1] if len(parts) > 1 else "?"

    results = []

    for dact in root.findall(".//dact"):
        # Get dialogue act type
        da_type_id = None
        for pointer in dact:
            tag = pointer.tag.split("}")[-1] if "}" in pointer.tag else pointer.tag
            if tag == "pointer" or "pointer" in pointer.tag:
                role = pointer.get("role", "")
                if role == "da-aspect":
                    da_type_id = get_da_type_id(pointer)

        if da_type_id not in TARGET_DA_IDS:
            continue

        # Get word references
        word_ids = []
        for child in dact:
            tag = child.tag.split("}")[-1] if "}" in child.tag else child.tag
            if tag == "child" or "child" in child.tag:
                href = child.get("href", "")
                words_file, start_id, end_id = parse_child_href(href)
                if words_file and start_id:
                    word_ids.append((words_file, start_id, end_id))

        # Resolve word IDs to timestamps and text
        words_file_key = f"{meeting_id}.{speaker}.words.xml"
        if words_file_key not in words_cache:
            words_path = WORDS_DIR / words_file_key
            if words_path.exists():
                words_cache[words_file_key] = parse_words_file(words_path)
            else:
                continue

        words_dict = words_cache[words_file_key]

        utterance_words = []
        start_time = None
        end_time = None

        for wf, sid, eid in word_ids:
            if eid is None:
                # Single word
                if sid in words_dict:
                    w = words_dict[sid]
                    utterance_words.append(w["text"])
                    if w["starttime"] is not None:
                        if start_time is None or w["starttime"] < start_time:
                            start_time = w["starttime"]
                    if w["endtime"] is not None:
                        if end_time is None or w["endtime"] > end_time:
                            end_time = w["endtime"]
            else:
                # Range of words: extract numeric suffixes and iterate
                prefix_match = re.match(r'(.+?)(\d+)$', sid)
                end_match = re.match(r'(.+?)(\d+)$', eid)
                if prefix_match and end_match:
                    prefix = prefix_match.group(1)
                    s_num = int(prefix_match.group(2))
                    e_num = int(end_match.group(2))
                    for i in range(s_num, e_num + 1):
                        wid = f"{prefix}{i}"
                        if wid in words_dict:
                            w = words_dict[wid]
                            utterance_words.append(w["text"])
                            if w["starttime"] is not None:
                                if start_time is None or w["starttime"] < start_time:
                                    start_time = w["starttime"]
                            if w["endtime"] is not None:
                                if end_time is None or w["endtime"] > end_time:
                                    end_time = w["endtime"]

        if start_time is not None and end_time is not None:
            text = " ".join(w for w in utterance_words if w and not w.startswith("["))
            da_label = "comment-about-understanding" if da_type_id == "ami_da_12" else "elicit-comment-understanding"

            dact_id_attr = None
            for k, v in dact.attrib.items():
                if k.endswith("}id") or k == "nite:id":
                    dact_id_attr = v
                    break

            results.append({
                "meeting_id": meeting_id,
                "speaker": speaker,
                "da_id": dact_id_attr,
                "da_type": da_label,
                "da_type_id": da_type_id,
                "start_time": round(start_time, 3),
                "end_time": round(end_time, 3),
                "duration_ms": round((end_time - start_time) * 1000, 1),
                "text": text.strip(),
            })

    return results


def main():
    words_cache = {}
    all_hdm = []

    da_files = sorted(DA_DIR.glob("*.dialog-act.xml"))
    print(f"Found {len(da_files)} dialogue act files")

    for da_file in da_files:
        hdms = parse_dialogue_acts(str(da_file), words_cache)
        all_hdm.extend(hdms)

    print(f"\nTotal Hearing Difficulty Moments found: {len(all_hdm)}")

    # Count by type
    type_counts = defaultdict(int)
    for h in all_hdm:
        type_counts[h["da_type"]] += 1
    for t, c in type_counts.items():
        print(f"  {t}: {c}")

    # Count by meeting
    meeting_counts = defaultdict(int)
    for h in all_hdm:
        meeting_counts[h["meeting_id"]] += 1
    print(f"\nMeetings with HDMs: {len(meeting_counts)}")

    # Duration statistics
    durations = [h["duration_ms"] for h in all_hdm]
    if durations:
        print(f"\nDuration stats (ms):")
        print(f"  Mean: {sum(durations)/len(durations):.1f}")
        print(f"  Min: {min(durations):.1f}")
        print(f"  Max: {max(durations):.1f}")
        print(f"  Median: {sorted(durations)[len(durations)//2]:.1f}")

    # Show some examples
    print(f"\nSample HDMs:")
    for h in all_hdm[:15]:
        print(f"  [{h['meeting_id']}.{h['speaker']}] {h['da_type']}: "
              f"\"{h['text']}\" ({h['start_time']:.2f}s - {h['end_time']:.2f}s, {h['duration_ms']:.0f}ms)")

    # Save to CSV
    output_path = Path("data/hdm_annotations.csv")
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_hdm[0].keys())
        writer.writeheader()
        writer.writerows(all_hdm)
    print(f"\nSaved {len(all_hdm)} HDMs to {output_path}")

    # Save to JSON too
    with open("data/hdm_annotations.json", "w") as f:
        json.dump(all_hdm, f, indent=2)

    return all_hdm


if __name__ == "__main__":
    main()
