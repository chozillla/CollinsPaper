"""
20-Shot Audio LLM Classifier v4 — balanced precision/recall.

Improvements over v3 (which had great precision but low recall):
- 20-shot (12 pos + 8 neg) — more positive examples to boost recall
- Mixed negatives: hard negatives + random negatives (not all hard)
- Balanced prompt: removed overly conservative warning
- 12s extended context + transcripts (same as v3)
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
N_SHOTS = 20  # 12 pos + 8 neg
N_POS_SHOTS = 12
N_NEG_SHOTS = 8

SYSTEM_PROMPT = """You are an expert at analyzing if a speaker in a given conversation is having difficulties understanding or hearing at a given moment. Please consider the following factors:

* **Non-semantic / acoustic information:** Listen carefully to the speaker's voice quality and prosody. The Lombard effect and related acoustic changes occur when someone is struggling to hear in conversation:
  - Increase in phonetic fundamental frequency (F0) — voice pitch rises
  - Shift in energy from low frequency bands to middle or high bands
  - Increase in sound intensity — speaking louder to compensate
  - Increase in vowel duration — stretching sounds out
  - Spectral tilting — flattening of spectral slope
  - Shift in formant center frequencies for F1 (mainly) and F2
  - Content words prolonged more than function words in noise
  - Greater lung volumes used — more effortful speech
  - Voice quality changes: strained, tense, or effortful phonation
  - Speaking rate changes: slowing down, hesitating, or pausing mid-utterance
  - Rising intonation with a confused or uncertain tone on short utterances
  - Filled pauses or hesitation markers (um, uh) before asking for repetition
  - Abrupt interruption of the conversational flow — a break in the natural turn-taking rhythm
* **Contextual acoustic cues:**
  - Background noise level — higher noise increases likelihood of hearing difficulty
  - Whether the preceding speech from other speakers was unclear, overlapping, or low volume
  - Contrast between the speaker's normal voice and their voice at the moment in question
* **Semantic information:** Pay attention to what they are saying and any keywords which might indicate that they are struggling to understand something. Are they asking for clarifications? Common examples to look out for (not exhaustive):
  - What?
  - Can you repeat that?
  - I didn't catch that?
  - Huh?
  - Sorry?
  - Pardon?
  - Say that again?
  - I missed that
  - Could you speak up?
* **Subjectivity:** Recognize that some experiences are inherently subjective. Focus on the speaker's experience rather than your personal opinions. Do you think they are having a moment of hearing difficulty?

You will hear an extended audio clip with a transcript. The potential hearing difficulty moment occurs in the MIDDLE portion. Use the audio before and after for context to judge whether the speaker is genuinely struggling to hear or understand.

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


def select_shot_examples_v4(train_indices, all_examples, extended_audio, labels_np,
                            human_labels, transcripts, n_pos=N_POS_SHOTS, n_neg=N_NEG_SHOTS):
    """Select 20-shot examples: 12 pos + 8 neg (mixed hard + random negatives)."""
    n_positives = 149

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

    # 12 positive examples: prefer human-verified, diverse text patterns
    pos_pool = verified_pos if len(verified_pos) >= n_pos else verified_pos + unverified_pos
    selected_pos = np.random.choice(pos_pool, size=min(n_pos, len(pos_pool)), replace=False)

    # 8 negative examples: use ALL hard negatives + fill rest with random
    n_hard = min(len(hard_neg), n_neg // 2)  # up to half from hard negatives
    n_random = n_neg - n_hard

    hard_selected = np.random.choice(hard_neg, size=n_hard, replace=False) if hard_neg else np.array([], dtype=int)
    random_selected = np.random.choice(random_neg, size=min(n_random, len(random_neg)), replace=False)
    selected_neg = np.concatenate([hard_selected, random_selected]).astype(int)

    # Build examples list: interleave pos and neg
    examples = []
    pos_list = list(selected_pos)
    neg_list = list(selected_neg)

    # Interleave: P, P, N, P, P, N, ... (2:1 ratio roughly)
    pi, ni = 0, 0
    while pi < len(pos_list) or ni < len(neg_list):
        # Add 2 positives
        for _ in range(2):
            if pi < len(pos_list):
                idx = pos_list[pi]
                if idx in extended_audio:
                    examples.append({
                        "audio": extended_audio[idx],
                        "label": 1,
                        "transcript": transcripts.get(str(idx), ""),
                    })
                pi += 1
        # Add 1 negative
        if ni < len(neg_list):
            idx = neg_list[ni]
            if idx in extended_audio:
                examples.append({
                    "audio": extended_audio[idx],
                    "label": 0,
                    "transcript": transcripts.get(str(idx), ""),
                })
            ni += 1

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
    shot_examples = select_shot_examples_v4(
        train_indices, all_examples, extended_audio, labels_np,
        human_labels, transcripts, N_POS_SHOTS, N_NEG_SHOTS
    )
    n_pos_shots = sum(1 for e in shot_examples if e["label"] == 1)
    n_neg_shots = sum(1 for e in shot_examples if e["label"] == 0)
    print(f"  {n_shots}-shot: {n_pos_shots} pos (human-verified), {n_neg_shots} neg (mixed hard+random)")

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

    print("GPT-4o Audio Classification v4")
    print("=" * 55)
    print("20-shot (12P+8N) + mixed negatives + balanced prompt")

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
    print(f"=== GPT-4o Audio v4 ({n_shots}-Shot, balanced) ===")
    print(f"Average F1: {avg_f1:.4f} (+/- {std_f1:.4f})")
    print(f"Per-split:  {[round(r['f1'], 4) for r in results]}")
    print(f"\nPaper (Gemini 1.5 Pro 10-shot): F1 = 0.87")
    print(f"v1 GPT-4o (4s):                    F1 = 0.1427")
    print(f"v2 GPT-4o (12s, human shots):       F1 = 0.4564")
    print(f"v3 GPT-4o (hard neg, strict):        F1 = 0.6000")

    output = {
        "method": f"GPT-4o Audio v4 {n_shots}-Shot (12s, transcripts, mixed neg, balanced prompt)",
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
    with open(RESULTS_DIR / f"gpt4o_{n_shots}shot_v4_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}/gpt4o_{n_shots}shot_v4_results.json")


if __name__ == "__main__":
    main()
