"""
10-Shot Audio LLM Classifier using Gemini 3.1 Pro

Replicating Collins et al. Section 2.3 — the paper used Gemini 1.5 Pro,
we use the newer Gemini 3.1 Pro which has the same native audio input capability.

Approach:
- Feed 4-second audio segments directly to Gemini with the paper's exact prompt
- 10-shot: 5 positive + 5 negative audio examples with labels
- Classify target segment as "P" (hearing difficulty) or "N" (no difficulty)
- Monte Carlo cross-validation (5 splits, 80/20)
"""

import json
import io
import os
import random
import numpy as np
import soundfile as sf
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
from tqdm import tqdm
import time

load_dotenv()

DATASET_DIR = Path("data/dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
RANDOM_SEED = 42
MODEL_NAME = "gemini-3.1-pro-preview"

# The exact prompt from Collins et al. (Section 2.3)
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

Use all of the context available but make your judgement only on if the current moment (ie. the end of the audio) is a hearing difficulty event or not.

Answer only with "P" for POSITIVE meaning a hearing difficulty event or "N" for NEGATIVE meaning it isn't a hearing difficulty event. Do not include any other rationale or fluff in your response."""


def get_client():
    """Create Gemini client."""
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def audio_to_wav_bytes(audio_array, sr=SAMPLE_RATE):
    """Convert numpy audio array to WAV bytes."""
    buf = io.BytesIO()
    sf.write(buf, audio_array.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    return buf.read()


def build_few_shot_contents(shot_examples, target_audio):
    """Build the content list for few-shot audio classification.

    Following the paper: present examples as
      (Audio: [audio], Label: P/N)
    then the target:
      (Audio: [audio], Label: )
    """
    contents = []

    # Few-shot examples as conversation turns
    for ex in shot_examples:
        wav_bytes = audio_to_wav_bytes(ex["audio"])
        label_str = "P" if ex["label"] == 1 else "N"

        # User turn: audio + "Label: "
        contents.append(
            types.Content(
                role="user",
                parts=[
                    types.Part.from_bytes(data=wav_bytes, mime_type="audio/wav"),
                    types.Part.from_text(text="Label: "),
                ],
            )
        )
        # Model turn: the label
        contents.append(
            types.Content(
                role="model",
                parts=[types.Part.from_text(text=label_str)],
            )
        )

    # Target audio for classification
    target_wav = audio_to_wav_bytes(target_audio)
    contents.append(
        types.Content(
            role="user",
            parts=[
                types.Part.from_bytes(data=target_wav, mime_type="audio/wav"),
                types.Part.from_text(text="Label: "),
            ],
        )
    )

    return contents


def classify_segment(client, shot_examples, target_audio):
    """Classify a single audio segment using 10-shot prompting."""
    contents = build_few_shot_contents(shot_examples, target_audio)

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                max_output_tokens=100,  # need >1 to accommodate thinking
                temperature=0,
            ),
        )

        # Extract non-thought text parts (Gemini 3.1 Pro has thinking by default)
        text_parts = []
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if not getattr(part, "thought", False) and part.text:
                    text_parts.append(part.text.strip())

        answer = "".join(text_parts).strip().upper()

        # Confidence: simple binary from the answer (logprobs not reliably
        # available with thinking models, so we use the answer directly)
        prob_p = 0.9 if answer == "P" else 0.1

        prediction = 1 if answer == "P" else 0
        return prediction, prob_p

    except Exception as e:
        print(f"    API error: {e}")
        time.sleep(5)
        return -1, 0.5


def select_shot_examples(train_audio, train_labels, n_shots=10):
    """Select n_shots examples (equal positive/negative) for few-shot."""
    n_per_class = n_shots // 2
    pos_indices = np.where(train_labels == 1)[0]
    neg_indices = np.where(train_labels == 0)[0]

    selected_pos = np.random.choice(
        pos_indices, size=min(n_per_class, len(pos_indices)), replace=False
    )
    selected_neg = np.random.choice(
        neg_indices, size=min(n_per_class, len(neg_indices)), replace=False
    )

    examples = []
    for p, n in zip(selected_pos, selected_neg):
        examples.append({"audio": train_audio[p], "label": 1})
        examples.append({"audio": train_audio[n], "label": 0})

    return examples


def evaluate_split(client, audio_segments, labels, meta, split_idx, n_shots=10):
    """Evaluate on one Monte Carlo CV split."""
    split = meta["splits"][split_idx]
    train_meetings = set(split["train"])
    test_meetings = set(split["test"])

    all_examples = meta["positive"] + meta["negative"]

    train_indices = [
        i for i, ex in enumerate(all_examples) if ex["meeting_id"] in train_meetings
    ]
    test_indices = [
        i for i, ex in enumerate(all_examples) if ex["meeting_id"] in test_meetings
    ]

    train_audio = audio_segments[train_indices]
    train_labels = labels[train_indices]
    test_audio = audio_segments[test_indices]
    test_labels = labels[test_indices]

    print(
        f"  Train: {len(train_indices)} ({train_labels.sum()} pos), "
        f"Test: {len(test_indices)} ({test_labels.sum()} pos)"
    )

    # Select few-shot examples from training set
    np.random.seed(RANDOM_SEED + split_idx)
    shot_examples = select_shot_examples(train_audio, train_labels, n_shots)
    n_pos_shots = sum(1 for e in shot_examples if e["label"] == 1)
    n_neg_shots = sum(1 for e in shot_examples if e["label"] == 0)
    print(f"  {n_shots}-shot examples: {n_pos_shots} pos, {n_neg_shots} neg")

    # Classify each test segment
    predictions = []
    probabilities = []
    errors = 0

    for i in tqdm(range(len(test_audio)), desc=f"  Split {split_idx + 1}"):
        pred, prob = classify_segment(client, shot_examples, test_audio[i])
        if pred == -1:
            errors += 1
            pred = 0  # default to negative on error
        predictions.append(pred)
        probabilities.append(prob)

        # Rate limit: Gemini free tier is 15 RPM, paid is higher
        time.sleep(1.0)

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    f1 = f1_score(test_labels, predictions, zero_division=0)

    precision, recall, thresholds = precision_recall_curve(test_labels, probabilities)

    print(f"  Errors: {errors}/{len(test_audio)}")

    return {
        "split": split_idx,
        "f1": f1,
        "n_shots": n_shots,
        "n_test": len(test_indices),
        "n_pos": int(test_labels.sum()),
        "n_pred_pos": int(predictions.sum()),
        "errors": errors,
        "report": classification_report(
            test_labels, predictions, output_dict=True, zero_division=0
        ),
        "precision_curve": precision.tolist(),
        "recall_curve": recall.tolist(),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "true_labels": test_labels.tolist(),
    }


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

    print(f"Gemini 3.1 Pro 10-Shot Audio Classification")
    print("=" * 55)
    print("Replicating Collins et al. Section 2.3")
    print(f"Paper: Gemini 1.5 Pro → Ours: {MODEL_NAME}")

    if not os.getenv("GEMINI_API_KEY") or "your-" in os.getenv("GEMINI_API_KEY", ""):
        print("\nERROR: Set your Gemini API key in .env (GEMINI_API_KEY=...)")
        return

    client = get_client()

    # Quick connectivity test
    print("\nTesting API connection...")
    try:
        test_response = client.models.generate_content(
            model=MODEL_NAME,
            contents="Say OK",
            config=types.GenerateContentConfig(max_output_tokens=2),
        )
        print(f"  API OK: {test_response.text.strip()}")
    except Exception as e:
        print(f"  API ERROR: {e}")
        print("  Check your GEMINI_API_KEY and model name.")
        return

    print("\nLoading dataset...")
    audio_segments = np.load(DATASET_DIR / "audio_segments.npy")
    labels = np.load(DATASET_DIR / "labels.npy")
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    print(f"Dataset: {len(audio_segments)} segments, {labels.sum()} positive")

    n_shots = 10
    results = []

    for split_idx in range(len(meta["splits"])):
        print(f"\n{'=' * 55}")
        print(f"Split {split_idx + 1}/{len(meta['splits'])} ({n_shots}-shot)")
        print(f"{'=' * 55}")

        result = evaluate_split(
            client, audio_segments, labels, meta, split_idx, n_shots
        )
        results.append(result)
        print(
            f"\n  F1: {result['f1']:.4f} "
            f"(pred_pos={result['n_pred_pos']}, true_pos={result['n_pos']})"
        )

    avg_f1 = np.mean([r["f1"] for r in results])
    std_f1 = np.std([r["f1"] for r in results])

    print(f"\n{'=' * 55}")
    print(f"=== {MODEL_NAME} {n_shots}-Shot Results ===")
    print(f"Average F1: {avg_f1:.4f} (+/- {std_f1:.4f})")
    print(f"Per-split:  {[round(r['f1'], 4) for r in results]}")
    print(f"\nPaper (Gemini 1.5 Pro {n_shots}-shot): F1 = 0.87")

    output = {
        "method": f"{MODEL_NAME} {n_shots}-Shot Audio Classification",
        "paper_method": f"Gemini 1.5 Pro {n_shots}-shot",
        "paper_f1": 0.87,
        "avg_f1": avg_f1,
        "std_f1": std_f1,
        "n_shots": n_shots,
        "model": MODEL_NAME,
        "prompt": SYSTEM_PROMPT,
        "splits": results,
    }
    with open(RESULTS_DIR / f"gemini_{n_shots}shot_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}/gemini_{n_shots}shot_results.json")


if __name__ == "__main__":
    main()
