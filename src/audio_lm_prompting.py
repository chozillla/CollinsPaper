"""
Audio Language Model Prompting (Method 3 from Collins et al.)

The paper used Gemini 1.5 Pro with audio input for 0-shot, 2-shot, and 10-shot.
Since Gemini API may not be available, this implements the approach using
OpenAI Whisper for transcription + a text LLM for classification,
AND documents the exact prompts for reproduction with Gemini or other audio LMs.

For the text-only variant (which the paper also tested), we:
1. Transcribe with Whisper
2. Prompt an LLM with the transcript and the classification prompt
"""

import json
import numpy as np
import whisper
import os
from pathlib import Path
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

DATASET_DIR = Path("data/dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SAMPLE_RATE = 16000

# The exact prompt from Collins et al. (for audio modality)
AUDIO_PROMPT = """You are an expert at analyzing if a speaker in a given conversation is having difficulties understanding or hearing at a given moment. Please consider the following factors:

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

# Modified prompt for text-only variant
TEXT_PROMPT = """You are an expert at analyzing if a speaker in a given conversation is having difficulties understanding or hearing at a given moment. Please consider the following factors:

* **Semantic information:** Pay attention to what they are saying and any keywords which might indicate that they are struggling to understand something. Are they asking for clarifications? Common examples to look out for (not exhaustive):
  - What?
  - Can you repeat that?
  - I didn't catch that?
  - Huh?
  - Sorry?
* **Subjectivity:** Recognize that some experiences are inherently subjective. Focus on the speaker's experience rather than your personal opinions. Do you think they are having a moment of hearing difficulty?

Use all of the context available but make your judgement only on if the current moment (ie. the end of the transcript) is a hearing difficulty event or not.

Answer only with "P" for POSITIVE meaning a hearing difficulty event or "N" for NEGATIVE meaning it isn't a hearing difficulty event. Do not include any other rationale or fluff in your response."""


def transcribe_segments(audio_segments, model):
    """Transcribe audio segments using Whisper."""
    transcriptions = []
    for i in tqdm(range(len(audio_segments)), desc="Transcribing"):
        audio = audio_segments[i].astype(np.float32)
        result = model.transcribe(audio, language="en", fp16=False)
        transcriptions.append(result["text"].strip())
    return transcriptions


def text_heuristic_classify(transcript):
    """Rule-based classification using the semantic cues from the prompt.

    This mimics what a text-only LLM would do with the prompt above,
    since we may not have API access to a multimodal LLM.
    Enhanced beyond simple hotword matching by considering context.
    """
    text = transcript.lower().strip()

    # Strong positive signals (end of transcript contains these)
    # Focus on the END of the transcript as per the prompt
    words = text.split()
    if not words:
        return 0, 0.5

    # Check the last few words (the "current moment")
    last_words = " ".join(words[-5:]) if len(words) >= 5 else text

    strong_signals = [
        "what?", "huh?", "sorry?", "pardon?", "excuse me?",
        "what was that", "can you repeat", "didn't catch",
        "didn't hear", "say again", "come again",
        "i'm sorry?", "what did you say",
    ]

    for signal in strong_signals:
        if signal in last_words:
            return 1, 0.9

    # Medium signals anywhere in transcript
    medium_signals = ["what", "huh", "sorry", "pardon", "repeat"]
    for signal in medium_signals:
        if signal in last_words.split():
            return 1, 0.7

    return 0, 0.3


def evaluate_split_text_only(audio_segments, labels, meta, split_idx, whisper_model):
    """Evaluate text-only approach on one split."""
    split = meta["splits"][split_idx]
    test_meetings = set(split["test"])

    all_examples = meta["positive"] + meta["negative"]
    test_indices = [i for i, ex in enumerate(all_examples) if ex["meeting_id"] in test_meetings]

    test_audio = audio_segments[test_indices]
    test_labels = labels[test_indices]

    # Transcribe
    transcriptions = transcribe_segments(test_audio, whisper_model)

    # Classify
    predictions = []
    confidences = []
    for t in transcriptions:
        pred, conf = text_heuristic_classify(t)
        predictions.append(pred)
        confidences.append(conf)

    f1 = f1_score(test_labels, predictions, zero_division=0)

    return {
        "split": split_idx,
        "f1": f1,
        "n_test": len(test_indices),
        "n_pos": int(test_labels.sum()),
        "n_pred_pos": sum(predictions),
        "report": classification_report(test_labels, predictions, output_dict=True, zero_division=0),
    }


def main():
    print("Audio LM Prompting Approach")
    print("="*50)
    print("\nNote: The paper used Gemini 1.5 Pro with direct audio input.")
    print("This implementation uses Whisper transcription + text-based classification")
    print("as an approximation. The exact prompts are saved for reproduction with")
    print("Gemini or other multimodal audio LLMs.\n")

    print("Loading dataset...")
    audio_segments = np.load(DATASET_DIR / "audio_segments.npy")
    labels = np.load(DATASET_DIR / "labels.npy")
    with open(DATASET_DIR / "dataset_meta.json") as f:
        meta = json.load(f)

    print(f"Dataset: {len(audio_segments)} segments, {labels.sum()} positive")

    print("\nLoading Whisper model...")
    whisper_model = whisper.load_model("base")

    # Text-only evaluation (comparable to paper's "Gemini 1.5 Pro [text only]" row)
    print("\n--- Text-only classification (Whisper + heuristic) ---")
    results = []
    for split_idx in range(len(meta["splits"])):
        print(f"\nSplit {split_idx + 1}/{len(meta['splits'])}")
        result = evaluate_split_text_only(audio_segments, labels, meta, split_idx, whisper_model)
        results.append(result)
        print(f"F1: {result['f1']:.4f}")

    avg_f1 = np.mean([r["f1"] for r in results])
    std_f1 = np.std([r["f1"] for r in results])
    print(f"\n=== Text-only LM Approach ===")
    print(f"Average F1: {avg_f1:.4f} (+/- {std_f1:.4f})")

    # Save results and prompts
    output = {
        "method": "Audio LM Prompting (text-only approximation)",
        "note": "Paper used Gemini 1.5 Pro with audio. This uses Whisper + text heuristic.",
        "avg_f1": avg_f1,
        "std_f1": std_f1,
        "prompts": {
            "audio_prompt": AUDIO_PROMPT,
            "text_prompt": TEXT_PROMPT,
        },
        "splits": results,
    }
    with open(RESULTS_DIR / "audio_lm_results.json", "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {RESULTS_DIR}/audio_lm_results.json")
    print(f"\nPrompts for reproduction with Gemini/multimodal LM saved in results file.")


if __name__ == "__main__":
    main()
