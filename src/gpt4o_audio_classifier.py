"""
10-Shot Audio LLM Classifier (replicating Collins et al. Section 2.3)

Uses Azure GPT-4o with native audio input to classify Hearing Difficulty Moments.

The paper's approach:
- Feed audio directly to multimodal LLM (they used Gemini 1.5 Pro)
- Prompt with detailed instructions about non-semantic and semantic cues
- Few-shot: provide N positive and N negative audio examples with labels
- Classify the target audio segment as "P" (positive) or "N" (negative)
- Use log probabilities of P vs N tokens for confidence scoring

We replicate with GPT-4o via Azure, which also supports native audio input.
"""

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
import time

load_dotenv()

DATASET_DIR = Path("data/dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000
RANDOM_SEED = 42

# --- The exact prompt from Collins et al. (Section 2.3) ---
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
    """Create Azure OpenAI client."""
    return AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2025-01-01-preview"),
    )


def audio_to_base64_wav(audio_array, sr=SAMPLE_RATE):
    """Convert numpy audio array to base64-encoded WAV."""
    buf = io.BytesIO()
    sf.write(buf, audio_array.astype(np.float32), sr, format="WAV")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def build_few_shot_messages(shot_examples, target_audio_b64):
    """Build the message list for few-shot audio classification.

    Following the paper: present N positive and N negative examples with labels,
    then the target audio for classification.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Add few-shot examples
    for ex in shot_examples:
        audio_b64 = audio_to_base64_wav(ex["audio"])
        label_str = "P" if ex["label"] == 1 else "N"

        # User message with audio
        messages.append({
            "role": "user",
            "content": [
                {
                    "type": "input_audio",
                    "input_audio": {
                        "data": audio_b64,
                        "format": "wav",
                    }
                },
                {
                    "type": "text",
                    "text": "Label: "
                }
            ]
        })
        # Assistant response with label
        messages.append({
            "role": "assistant",
            "content": label_str
        })

    # Add target audio for classification
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "input_audio",
                "input_audio": {
                    "data": target_audio_b64,
                    "format": "wav",
                }
            },
            {
                "type": "text",
                "text": "Label: "
            }
        ]
    })

    return messages


def classify_segment(client, deployment, shot_examples, target_audio):
    """Classify a single audio segment using the 10-shot approach."""
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

        # Extract log probabilities for confidence scoring
        prob_p = 0.5  # default
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
                # Softmax over P and N
                import math
                max_log = max(log_p, log_n)
                prob_p = math.exp(log_p - max_log) / (math.exp(log_p - max_log) + math.exp(log_n - max_log))

        prediction = 1 if answer == "P" else 0
        return prediction, prob_p

    except Exception as e:
        print(f"    API error: {e}")
        return 0, 0.5


def select_shot_examples(train_audio, train_labels, n_shots=10):
    """Select n_shots examples (equal positive/negative) for few-shot prompting.

    Following the paper: equal number of positive and negative examples,
    randomly drawn.
    """
    n_per_class = n_shots // 2
    pos_indices = np.where(train_labels == 1)[0]
    neg_indices = np.where(train_labels == 0)[0]

    selected_pos = np.random.choice(pos_indices, size=min(n_per_class, len(pos_indices)), replace=False)
    selected_neg = np.random.choice(neg_indices, size=min(n_per_class, len(neg_indices)), replace=False)

    examples = []
    # Interleave positive and negative
    for p, n in zip(selected_pos, selected_neg):
        examples.append({"audio": train_audio[p], "label": 1})
        examples.append({"audio": train_audio[n], "label": 0})

    # Add any remaining
    for idx in selected_pos[len(selected_neg):]:
        examples.append({"audio": train_audio[idx], "label": 1})
    for idx in selected_neg[len(selected_pos):]:
        examples.append({"audio": train_audio[idx], "label": 0})

    return examples


def evaluate_split(client, deployment, audio_segments, labels, meta, split_idx, n_shots=10):
    """Evaluate on one Monte Carlo CV split."""
    split = meta["splits"][split_idx]
    train_meetings = set(split["train"])
    test_meetings = set(split["test"])

    all_examples = meta["positive"] + meta["negative"]

    train_indices = [i for i, ex in enumerate(all_examples) if ex["meeting_id"] in train_meetings]
    test_indices = [i for i, ex in enumerate(all_examples) if ex["meeting_id"] in test_meetings]

    train_audio = audio_segments[train_indices]
    train_labels = labels[train_indices]
    test_audio = audio_segments[test_indices]
    test_labels = labels[test_indices]

    print(f"  Train: {len(train_indices)} ({train_labels.sum()} pos), Test: {len(test_indices)} ({test_labels.sum()} pos)")

    # Select few-shot examples from training set
    np.random.seed(RANDOM_SEED + split_idx)
    shot_examples = select_shot_examples(train_audio, train_labels, n_shots)
    print(f"  Selected {len(shot_examples)} shot examples ({sum(1 for e in shot_examples if e['label']==1)} pos, {sum(1 for e in shot_examples if e['label']==0)} neg)")

    # Classify each test segment
    predictions = []
    probabilities = []

    for i in tqdm(range(len(test_audio)), desc=f"  Classifying split {split_idx+1}"):
        pred, prob = classify_segment(client, deployment, shot_examples, test_audio[i])
        predictions.append(pred)
        probabilities.append(prob)

        # Rate limiting — be conservative
        time.sleep(0.5)

    predictions = np.array(predictions)
    probabilities = np.array(probabilities)

    f1 = f1_score(test_labels, predictions, zero_division=0)

    # Precision-recall curve
    precision, recall, thresholds = precision_recall_curve(test_labels, probabilities)

    return {
        "split": split_idx,
        "f1": f1,
        "n_shots": n_shots,
        "n_test": len(test_indices),
        "n_pos": int(test_labels.sum()),
        "n_pred_pos": int(predictions.sum()),
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

    print("GPT-4o 10-Shot Audio Classification")
    print("=" * 50)
    print("Replicating Collins et al. Section 2.3 (Gemini 1.5 Pro → GPT-4o)")

    # Check API credentials
    if not os.getenv("AZURE_OPENAI_API_KEY") or os.getenv("AZURE_OPENAI_API_KEY") == "your-key-here":
        print("\nERROR: Set your Azure OpenAI API key in .env")
        return

    client = get_client()
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")

    print(f"\nEndpoint: {os.getenv('AZURE_OPENAI_ENDPOINT')}")
    print(f"Deployment: {deployment}")

    print("\nLoading dataset...")
    audio_segments = np.load(DATASET_DIR / "audio_segments.npy")
    labels = np.load(DATASET_DIR / "labels.npy")
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    print(f"Dataset: {len(audio_segments)} segments, {labels.sum()} positive")

    # Run 10-shot evaluation across all splits
    n_shots = 10
    results = []

    for split_idx in range(len(meta["splits"])):
        print(f"\n{'='*50}")
        print(f"Split {split_idx + 1}/{len(meta['splits'])} ({n_shots}-shot)")
        print(f"{'='*50}")

        result = evaluate_split(client, deployment, audio_segments, labels, meta, split_idx, n_shots)
        results.append(result)
        print(f"\n  F1: {result['f1']:.4f} (pred_pos={result['n_pred_pos']}, true_pos={result['n_pos']})")

    avg_f1 = np.mean([r["f1"] for r in results])
    std_f1 = np.std([r["f1"] for r in results])
    print(f"\n{'='*50}")
    print(f"=== GPT-4o {n_shots}-Shot Audio Classification ===")
    print(f"Average F1: {avg_f1:.4f} (+/- {std_f1:.4f})")
    print(f"Per-split: {[round(r['f1'], 4) for r in results]}")
    print(f"\nPaper's Gemini 1.5 Pro {n_shots}-shot: F1 = 0.87")

    # Save results
    output = {
        "method": f"GPT-4o {n_shots}-Shot Audio Classification (Azure)",
        "paper_method": f"Gemini 1.5 Pro {n_shots}-shot",
        "paper_f1": 0.87,
        "avg_f1": avg_f1,
        "std_f1": std_f1,
        "n_shots": n_shots,
        "prompt": SYSTEM_PROMPT,
        "splits": [{k: v for k, v in r.items()} for r in results],
    }
    with open(RESULTS_DIR / f"gpt4o_{n_shots}shot_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}/gpt4o_{n_shots}shot_results.json")


if __name__ == "__main__":
    main()
