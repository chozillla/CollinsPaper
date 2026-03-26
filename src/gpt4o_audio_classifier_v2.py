"""
10-Shot Audio LLM Classifier v2 — with human-verified labels and extended context.

Improvements over v1:
- Uses human-verified HDM labels for 10-shot example selection
- Sends 12 seconds of audio per segment (4s before + 4s HDM + 4s after)
  instead of just the 4s HDM segment, giving the model more context
- Uses GPT-4o-audio on Azure with real logprobs
"""

import gc
import json
import base64
import io
import os
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from openai import AzureOpenAI
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

load_dotenv()

AUDIO_DIR = Path("data/audio")
DATASET_DIR = Path("data/dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
RANDOM_SEED = 42
CONTEXT_BEFORE = 4.0  # seconds before the HDM segment
CONTEXT_AFTER = 4.0   # seconds after the HDM segment
SEGMENT_DURATION = 4.0
MAX_WORKERS = 20

SYSTEM_PROMPT = """You are an expert at analyzing if a speaker in a given conversation is having difficulties understanding or hearing at a given moment. Please consider the following factors:

* **Non-semantic information:** Assess the general tone and pitch expressed in the speakers voice. Is it predominantly strained, or at ease? The lombard effect describes the things to look out for when someone is struggling in conversation:
  - increase in phonetic fundamental frequencies
  - shift in energy from low frequency bands to middle or high bands
  - increase in sound intensity
  - increase in vowel duration
  - spectral tilting
  - shift in formant center frequencies for F1 (mainly) and F2
  - the duration of content words are prolonged to a greater degree in noise than function words
  - greater lung volumes are used
* **Semantic information:** Pay attention to what they are saying and any keywords which might indicate that they are struggling to understand something. Are they asking for clarifications? Common examples to look out for (not exhaustive):
  - What?
  - Can you repeat that?
  - I didn't catch that?
  - Huh?
  - Sorry?
* **Subjectivity:** Recognize that some experiences are inherently subjective. Focus on the speaker's experience rather than your personal opinions. Do you think they are having a moment of hearing difficulty?

You will hear an extended audio clip. The hearing difficulty moment (if any) occurs in the MIDDLE portion of the clip. Use the audio before and after for context, but judge whether the middle portion contains a hearing difficulty event.

Answer only with "P" for POSITIVE meaning a hearing difficulty event or "N" for NEGATIVE meaning it isn't a hearing difficulty event. Do not include any other rationale or fluff in your response."""


def get_client():
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    )


def audio_to_base64_wav(audio_array, sr=SAMPLE_RATE):
    buf = io.BytesIO()
    sf.write(buf, audio_array.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def load_meeting_audio(meeting_id):
    audio_path = AUDIO_DIR / f"{meeting_id}.Mix-Headset.wav"
    if not audio_path.exists():
        return None
    data, sr = sf.read(str(audio_path), dtype="float32")
    if len(data.shape) > 1:
        data = data.mean(axis=1)
    return data


def extract_extended_segment(audio_data, sample_time):
    """Extract 12s clip: 4s before + 4s HDM + 4s after centered on sample_time."""
    sr = SAMPLE_RATE
    # HDM segment ends at sample_time, starts at sample_time - 4s
    hdm_end = sample_time
    hdm_start = sample_time - SEGMENT_DURATION

    # Extended clip
    clip_start = hdm_start - CONTEXT_BEFORE
    clip_end = hdm_end + CONTEXT_AFTER

    start_sample = int(clip_start * sr)
    end_sample = int(clip_end * sr)

    if start_sample < 0 or end_sample > len(audio_data):
        # Fall back to just the HDM segment if extended is out of bounds
        start_sample = max(0, int(hdm_start * sr))
        end_sample = min(len(audio_data), int(hdm_end * sr))

    return audio_data[start_sample:end_sample]


def prepare_extended_dataset():
    """Build extended audio segments (12s) from original meeting WAVs."""
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    all_examples = meta["positive"] + meta["negative"]

    # Load human labels
    human_labels = {}
    labels_path = Path("data/hdm_labels.json")
    if labels_path.exists():
        with open(labels_path) as f:
            human_labels = json.load(f)
        print(f"  Human labels loaded: {len(human_labels)} ({sum(1 for v in human_labels.values() if v=='yes')} yes, {sum(1 for v in human_labels.values() if v=='no')} no)")

    # Group all examples by meeting
    by_meeting = defaultdict(list)
    for i, ex in enumerate(all_examples):
        by_meeting[ex["meeting_id"]].append((i, ex))

    # Extract extended audio for each example
    extended_audio = {}  # index -> audio array
    print("  Extracting extended audio clips...")

    for mid, items in sorted(by_meeting.items()):
        audio = load_meeting_audio(mid)
        if audio is None:
            continue

        for i, ex in items:
            clip = extract_extended_segment(audio, ex["sample_time"])
            extended_audio[i] = clip

        del audio
        gc.collect()

    print(f"  Extracted {len(extended_audio)} extended clips")
    return meta, all_examples, extended_audio, human_labels


def build_few_shot_messages(shot_examples, target_audio_b64):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in shot_examples:
        audio_b64 = audio_to_base64_wav(ex["audio"])
        label_str = "P" if ex["label"] == 1 else "N"

        messages.append({
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                {"type": "text", "text": "Label: "}
            ]
        })
        messages.append({"role": "assistant", "content": label_str})

    messages.append({
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": target_audio_b64, "format": "wav"}},
            {"type": "text", "text": "Label: "}
        ]
    })

    return messages


def classify_segment(client, deployment, shot_examples, target_audio):
    target_b64 = audio_to_base64_wav(target_audio)
    messages = build_few_shot_messages(shot_examples, target_b64)

    try:
        response = client.chat.completions.create(
            model=deployment,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=5,
        )

        choice = response.choices[0]
        answer = choice.message.content.strip().upper()

        prob_p = 0.5
        if choice.logprobs and choice.logprobs.content:
            token_logprobs = choice.logprobs.content[0].top_logprobs
            log_p = None
            log_n = None
            for tlp in token_logprobs:
                tok = tlp.token.strip().upper()
                if tok == "P":
                    log_p = tlp.logprob
                elif tok == "N":
                    log_n = tlp.logprob
            if log_p is not None and log_n is not None:
                import math
                max_log = max(log_p, log_n)
                prob_p = math.exp(log_p - max_log) / (math.exp(log_p - max_log) + math.exp(log_n - max_log))

        prediction = 1 if answer == "P" else 0
        return prediction, prob_p

    except Exception as e:
        print(f"    API error: {e}")
        return -1, 0.5


def select_shot_examples_v2(train_indices, all_examples, extended_audio, labels, human_labels, n_shots=10):
    """Select 10-shot examples, preferring human-verified ones."""
    n_per_class = n_shots // 2

    # Separate human-verified positive/negative from training set
    verified_pos = []
    verified_neg_from_pos = []  # human said "no" to a regex-positive
    unverified_pos = []
    neg_indices = []

    n_positives = 149  # first 149 are positives in the dataset

    for idx in train_indices:
        if idx < n_positives:
            # It's a regex-positive
            human = human_labels.get(str(idx))
            if human == "yes":
                verified_pos.append(idx)
            elif human == "no":
                verified_neg_from_pos.append(idx)
            else:
                unverified_pos.append(idx)
        else:
            neg_indices.append(idx)

    # Pick positive examples: prefer human-verified
    pos_pool = verified_pos if len(verified_pos) >= n_per_class else verified_pos + unverified_pos
    selected_pos = np.random.choice(pos_pool, size=min(n_per_class, len(pos_pool)), replace=False)

    # Pick negative examples: include human-rejected HDMs + random negatives
    neg_pool = verified_neg_from_pos + neg_indices
    selected_neg = np.random.choice(neg_pool, size=min(n_per_class, len(neg_pool)), replace=False)

    examples = []
    for p, n in zip(selected_pos, selected_neg):
        if p in extended_audio:
            examples.append({"audio": extended_audio[p], "label": 1})
        if n in extended_audio:
            examples.append({"audio": extended_audio[n], "label": 0})

    return examples


def evaluate_split(client, deployment, all_examples, extended_audio, labels_np, meta,
                   human_labels, split_idx, n_shots=10):
    split = meta["splits"][split_idx]
    train_meetings = set(split["train"])
    test_meetings = set(split["test"])

    train_indices = [i for i, ex in enumerate(all_examples) if ex["meeting_id"] in train_meetings]
    test_indices = [i for i, ex in enumerate(all_examples) if ex["meeting_id"] in test_meetings]

    train_labels = labels_np[train_indices]
    test_labels = labels_np[test_indices]

    print(f"  Train: {len(train_indices)} ({train_labels.sum()} pos), Test: {len(test_indices)} ({test_labels.sum()} pos)")

    # Select few-shot examples with human verification preference
    np.random.seed(RANDOM_SEED + split_idx)
    shot_examples = select_shot_examples_v2(
        train_indices, all_examples, extended_audio, labels_np, human_labels, n_shots
    )
    n_pos_shots = sum(1 for e in shot_examples if e["label"] == 1)
    n_neg_shots = sum(1 for e in shot_examples if e["label"] == 0)
    print(f"  10-shot examples: {n_pos_shots} pos (human-verified), {n_neg_shots} neg")

    # Classify test segments in parallel
    results_map = {}
    errors = 0
    pbar = tqdm(total=len(test_indices), desc=f"  Split {split_idx+1}")

    # Filter to only test indices that have extended audio
    valid_test = [(i, idx) for i, idx in enumerate(test_indices) if idx in extended_audio]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(classify_segment, client, deployment, shot_examples, extended_audio[idx]): i
            for i, idx in valid_test
        }
        for future in as_completed(futures):
            local_i = futures[future]
            pred, prob = future.result()
            if pred == -1:
                errors += 1
                pred = 0
            results_map[local_i] = (pred, prob)
            pbar.update(1)

    pbar.close()

    predictions = np.array([results_map.get(i, (0, 0.5))[0] for i in range(len(test_indices))])
    probabilities = np.array([results_map.get(i, (0, 0.5))[1] for i in range(len(test_indices))])

    f1 = f1_score(test_labels, predictions, zero_division=0)
    precision, recall, thresholds = precision_recall_curve(test_labels, probabilities)

    print(f"  Errors: {errors}/{len(test_indices)}")

    return {
        "split": split_idx,
        "f1": f1,
        "n_shots": n_shots,
        "n_test": len(test_indices),
        "n_pos": int(test_labels.sum()),
        "n_pred_pos": int(predictions.sum()),
        "errors": errors,
        "report": classification_report(test_labels, predictions, output_dict=True, zero_division=0),
        "precision_curve": precision.tolist(),
        "recall_curve": recall.tolist(),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "true_labels": test_labels.tolist(),
    }


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print("GPT-4o Audio Classification v2")
    print("=" * 55)
    print("Extended context (12s) + human-verified 10-shot examples")

    if not os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") == "your-key-here":
        print("\nERROR: Set your Azure OpenAI API key in .env")
        return

    client = get_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-audio")

    print(f"\nEndpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"Deployment: {deployment}")

    print("\nPreparing dataset with extended audio...")
    meta, all_examples, extended_audio, human_labels = prepare_extended_dataset()

    labels_np = np.load(DATASET_DIR / "labels.npy")
    print(f"Dataset: {len(all_examples)} segments, {labels_np.sum()} positive")

    print("\nAPI connection: using gpt-audio (audio-only model, skipping text test)")

    n_shots = 10
    results = []

    for split_idx in range(len(meta["splits"])):
        print(f"\n{'='*55}")
        print(f"Split {split_idx + 1}/{len(meta['splits'])} ({n_shots}-shot, 12s context)")
        print(f"{'='*55}")

        result = evaluate_split(
            client, deployment, all_examples, extended_audio, labels_np,
            meta, human_labels, split_idx, n_shots
        )
        results.append(result)
        print(f"\n  F1: {result['f1']:.4f} (pred_pos={result['n_pred_pos']}, true_pos={result['n_pos']})")

    avg_f1 = np.mean([r["f1"] for r in results])
    std_f1 = np.std([r["f1"] for r in results])
    print(f"\n{'='*55}")
    print(f"=== GPT-4o Audio v2 ({n_shots}-Shot, 12s context) ===")
    print(f"Average F1: {avg_f1:.4f} (+/- {std_f1:.4f})")
    print(f"Per-split:  {[round(r['f1'], 4) for r in results]}")
    print(f"\nPaper (Gemini 1.5 Pro 10-shot): F1 = 0.87")
    print(f"v1 GPT-4o (4s, no human labels): F1 = 0.1427")

    output = {
        "method": f"GPT-4o Audio v2 {n_shots}-Shot (12s context, human-verified examples)",
        "paper_method": f"Gemini 1.5 Pro {n_shots}-shot",
        "paper_f1": 0.87,
        "avg_f1": avg_f1,
        "std_f1": std_f1,
        "n_shots": n_shots,
        "context_seconds": CONTEXT_BEFORE + SEGMENT_DURATION + CONTEXT_AFTER,
        "human_labels_used": len(human_labels),
        "prompt": SYSTEM_PROMPT,
        "splits": [{k: v for k, v in r.items()} for r in results],
    }
    with open(RESULTS_DIR / f"gpt4o_{n_shots}shot_v2_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}/gpt4o_{n_shots}shot_v2_results.json")


if __name__ == "__main__":
    main()
