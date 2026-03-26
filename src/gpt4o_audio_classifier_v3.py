"""
10-Shot Audio LLM Classifier v3 — maximum performance.

Improvements over v2:
- Hard negatives: human-rejected HDMs as negative examples (teaches model
  what looks like an HDM but isn't — e.g. "What?" used conversationally)
- Whisper transcripts: each audio clip comes with a transcript so the model
  has both acoustic AND semantic information
- Stricter prompt: emphasizes only classifying clear hearing difficulty
- 12s extended context (same as v2)
- Real logprobs from GPT-4o-audio on Azure
"""

import gc
import json
import base64
import io
import os
import random
import math
import numpy as np
import soundfile as sf
# whisper removed — use existing metadata transcripts
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
CONTEXT_BEFORE = 4.0
CONTEXT_AFTER = 4.0
SEGMENT_DURATION = 4.0
MAX_WORKERS = 20
N_SHOTS = 10  # 5 pos + 5 neg

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

IMPORTANT: Not every question or "What?" is a hearing difficulty moment. People say "What?" conversationally to express surprise, seek clarification on meaning (not hearing), or as a filler. Only classify as POSITIVE if the speaker genuinely could not hear or understand what was said due to auditory difficulty.

You will hear an extended audio clip with a transcript. The potential hearing difficulty moment occurs in the MIDDLE portion. Use all context to judge.

Answer only with "P" for POSITIVE (genuine hearing difficulty) or "N" for NEGATIVE (not hearing difficulty). Do not include any other text."""


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
    sr = SAMPLE_RATE
    hdm_end = sample_time
    hdm_start = sample_time - SEGMENT_DURATION
    clip_start = hdm_start - CONTEXT_BEFORE
    clip_end = hdm_end + CONTEXT_AFTER

    start_sample = int(clip_start * sr)
    end_sample = int(clip_end * sr)

    if start_sample < 0 or end_sample > len(audio_data):
        start_sample = max(0, int(hdm_start * sr))
        end_sample = min(len(audio_data), int(hdm_end * sr))

    return audio_data[start_sample:end_sample]


def build_transcripts(all_examples):
    """Use existing metadata transcripts — no Whisper needed."""
    transcripts = {}
    for i, ex in enumerate(all_examples):
        transcripts[str(i)] = ex.get("text", "")
    n_with_text = sum(1 for t in transcripts.values() if t)
    print(f"  Transcripts from metadata: {n_with_text} with text, {len(transcripts) - n_with_text} empty")
    return transcripts


def prepare_dataset():
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    all_examples = meta["positive"] + meta["negative"]

    human_labels = {}
    labels_path = Path("data/hdm_labels.json")
    if labels_path.exists():
        with open(labels_path) as f:
            human_labels = json.load(f)
        yes_count = sum(1 for v in human_labels.values() if v == "yes")
        no_count = sum(1 for v in human_labels.values() if v == "no")
        print(f"  Human labels: {len(human_labels)} ({yes_count} yes, {no_count} no)")

    by_meeting = defaultdict(list)
    for i, ex in enumerate(all_examples):
        by_meeting[ex["meeting_id"]].append((i, ex))

    extended_audio = {}
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


def build_few_shot_messages(shot_examples, target_audio_b64, target_transcript):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    for ex in shot_examples:
        audio_b64 = audio_to_base64_wav(ex["audio"])
        label_str = "P" if ex["label"] == 1 else "N"
        transcript = ex.get("transcript", "")

        messages.append({
            "role": "user",
            "content": [
                {"type": "input_audio", "input_audio": {"data": audio_b64, "format": "wav"}},
                {"type": "text", "text": f"Transcript: \"{transcript}\"\nLabel: "}
            ]
        })
        messages.append({"role": "assistant", "content": label_str})

    messages.append({
        "role": "user",
        "content": [
            {"type": "input_audio", "input_audio": {"data": target_audio_b64, "format": "wav"}},
            {"type": "text", "text": f"Transcript: \"{target_transcript}\"\nLabel: "}
        ]
    })

    return messages


def classify_segment(client, deployment, shot_examples, target_audio, target_transcript):
    target_b64 = audio_to_base64_wav(target_audio)
    messages = build_few_shot_messages(shot_examples, target_b64, target_transcript)

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
                max_log = max(log_p, log_n)
                prob_p = math.exp(log_p - max_log) / (math.exp(log_p - max_log) + math.exp(log_n - max_log))

        prediction = 1 if answer == "P" else 0
        return prediction, prob_p

    except Exception as e:
        print(f"    API error: {e}")
        return -1, 0.5


def select_shot_examples_v3(train_indices, all_examples, extended_audio, labels_np,
                            human_labels, transcripts, n_shots=N_SHOTS):
    """Select few-shot examples with hard negatives."""
    n_per_class = n_shots // 2
    n_positives = 149

    # Categorize training examples
    verified_pos = []
    hard_neg = []  # human said "no" to regex-positive — best negatives
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

    # Positive examples: prefer human-verified
    pos_pool = verified_pos if len(verified_pos) >= n_per_class else verified_pos + unverified_pos
    selected_pos = np.random.choice(pos_pool, size=min(n_per_class, len(pos_pool)), replace=False)

    # Negative examples: prefer hard negatives, fill rest with random
    if len(hard_neg) >= n_per_class:
        selected_neg = np.random.choice(hard_neg, size=n_per_class, replace=False)
    else:
        # Use all hard negatives + fill with random
        remaining = n_per_class - len(hard_neg)
        extra_neg = np.random.choice(random_neg, size=min(remaining, len(random_neg)), replace=False)
        selected_neg = np.array(list(hard_neg) + list(extra_neg))

    examples = []
    for p, n in zip(selected_pos, selected_neg):
        if p in extended_audio:
            examples.append({
                "audio": extended_audio[p],
                "label": 1,
                "transcript": transcripts.get(str(p), ""),
            })
        if n in extended_audio:
            examples.append({
                "audio": extended_audio[n],
                "label": 0,
                "transcript": transcripts.get(str(n), ""),
            })

    return examples


def evaluate_split(client, deployment, all_examples, extended_audio, labels_np,
                   meta, human_labels, transcripts, split_idx, n_shots=N_SHOTS):
    split = meta["splits"][split_idx]
    train_meetings = set(split["train"])
    test_meetings = set(split["test"])

    train_indices = [i for i, ex in enumerate(all_examples) if ex["meeting_id"] in train_meetings]
    test_indices = [i for i, ex in enumerate(all_examples) if ex["meeting_id"] in test_meetings]

    train_labels = labels_np[train_indices]
    test_labels = labels_np[test_indices]

    print(f"  Train: {len(train_indices)} ({train_labels.sum()} pos), Test: {len(test_indices)} ({test_labels.sum()} pos)")

    np.random.seed(RANDOM_SEED + split_idx)
    shot_examples = select_shot_examples_v3(
        train_indices, all_examples, extended_audio, labels_np,
        human_labels, transcripts, n_shots
    )
    n_pos_shots = sum(1 for e in shot_examples if e["label"] == 1)
    n_neg_shots = sum(1 for e in shot_examples if e["label"] == 0)
    print(f"  {n_shots}-shot: {n_pos_shots} pos (human-verified), {n_neg_shots} neg (hard negatives)")

    # Show what examples were selected
    for ex in shot_examples:
        label = "P" if ex["label"] == 1 else "N"
        t = ex["transcript"][:60] if ex["transcript"] else "(no transcript)"
        print(f"    [{label}] {t}")

    results_map = {}
    errors = 0
    pbar = tqdm(total=len(test_indices), desc=f"  Split {split_idx+1}")

    valid_test = [(i, idx) for i, idx in enumerate(test_indices) if idx in extended_audio]

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                classify_segment, client, deployment, shot_examples,
                extended_audio[idx], transcripts.get(str(idx), "")
            ): i
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

    tp = ((predictions == 1) & (test_labels == 1)).sum()
    fp = ((predictions == 1) & (test_labels == 0)).sum()
    fn = ((predictions == 0) & (test_labels == 1)).sum()
    print(f"  Errors: {errors}/{len(test_indices)}")
    print(f"  TP={tp} FP={fp} FN={fn}")

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

    print("GPT-4o Audio Classification v3")
    print("=" * 55)
    print("Hard negatives + Whisper transcripts + stricter prompt")

    if not os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") == "your-key-here":
        print("\nERROR: Set your Azure OpenAI API key in .env")
        return

    client = get_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-audio")

    print(f"\nEndpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"Deployment: {deployment}")

    print("\nPreparing dataset...")
    meta, all_examples, extended_audio, human_labels = prepare_dataset()

    labels_np = np.load(DATASET_DIR / "labels.npy")
    print(f"Dataset: {len(all_examples)} segments, {labels_np.sum()} positive")

    print("\nBuilding transcripts from metadata...")
    transcripts = build_transcripts(all_examples)

    n_shots = N_SHOTS
    results = []

    for split_idx in range(len(meta["splits"])):
        print(f"\n{'='*55}")
        print(f"Split {split_idx + 1}/{len(meta['splits'])} ({n_shots}-shot, 12s, transcripts)")
        print(f"{'='*55}")

        result = evaluate_split(
            client, deployment, all_examples, extended_audio, labels_np,
            meta, human_labels, transcripts, split_idx, n_shots
        )
        results.append(result)
        print(f"\n  F1: {result['f1']:.4f} (pred_pos={result['n_pred_pos']}, true_pos={result['n_pos']})")

    avg_f1 = np.mean([r["f1"] for r in results])
    std_f1 = np.std([r["f1"] for r in results])
    print(f"\n{'='*55}")
    print(f"=== GPT-4o Audio v3 ({n_shots}-Shot, transcripts, hard neg) ===")
    print(f"Average F1: {avg_f1:.4f} (+/- {std_f1:.4f})")
    print(f"Per-split:  {[round(r['f1'], 4) for r in results]}")
    print(f"\nPaper (Gemini 1.5 Pro 10-shot): F1 = 0.87")
    print(f"v1 GPT-4o (4s, no human labels):  F1 = 0.1427")
    print(f"v2 GPT-4o (12s, human-verified):   F1 = 0.4564")

    output = {
        "method": f"GPT-4o Audio v3 {n_shots}-Shot (12s, transcripts, hard negatives, strict prompt)",
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
    with open(RESULTS_DIR / f"gpt4o_{n_shots}shot_v3_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}/gpt4o_{n_shots}shot_v3_results.json")


if __name__ == "__main__":
    main()
