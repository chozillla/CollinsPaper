"""
Filter AMI Comment-About-Understanding annotations to retain only
genuine "hearing difficulty" moments (signal-non-understanding).

The AMI tag covers both understanding AND non-understanding.
Collins et al. filtered 522 → 298 from SWDA/MRDA by removing utterances
that weren't about hearing difficulty. We do the same here.

Strategy:
1. Keep short utterances with non-understanding keywords ("what", "sorry", "pardon", "huh", etc.)
2. Remove confirmations of understanding ("yeah", "okay", "right", "ah", "uh-huh")
3. Remove long restatements/explanations (likely confirming understanding)
4. Flag ambiguous cases for review
"""

import csv
import json
import re
from pathlib import Path
from collections import Counter


def is_hearing_difficulty(text, duration_ms, da_type):
    """Classify whether an utterance represents genuine hearing difficulty.

    Following Collins et al.:
    - Keep: "signal-non-understanding" — speaker didn't HEAR/CATCH what was said
      e.g. "What?", "Huh?", "Sorry?", "Can you repeat that?"
    - Remove: semantic clarification — speaker HEARD but asks about meaning/intent
      e.g. "What do you mean?", "How high is it?", "Where is the controller?"
    - Remove: confirmations of understanding — "Hmm.", "Okay", "Ah."
    - Must have '?' to be a non-understanding signal (period = acknowledgment)
    """
    text_lower = text.lower().strip()
    text_clean = text_lower.rstrip(".?!,; ")

    # CRITICAL: signals must be QUESTIONS (have '?') to indicate non-understanding.
    # "Hmm." / "sorry." with a period are acknowledgments or apologies, not HDMs.
    has_question = "?" in text

    # Strong positive: classic non-understanding signals WITH question mark
    strong_positive_patterns = [
        r"^\s*what\s*\?\s*$",           # "What?"
        r"^\s*huh\s*\?\s*$",            # "Huh?"
        r"^\s*hmm?\s*\?\s*$",           # "Hmm?" / "Hm?"
        r"^\s*sorry\s*\?\s*$",          # "Sorry?"
        r"^\s*pardon\s*\?\s*$",         # "Pardon?"
        r"^\s*excuse me\s*\?\s*$",      # "Excuse me?"
        r"^\s*come again\s*\?\s*$",     # "Come again?"
    ]

    for pattern in strong_positive_patterns:
        if re.search(pattern, text_lower):
            return "positive", "strong_keyword_match"

    # Explicit non-understanding phrases (regardless of punctuation)
    explicit_non_understanding = [
        r"^what did you say",
        r"^what was that",
        r"^what'?s that\s*\?",
        r"^which was that",
        r"^can you repeat",
        r"^could you repeat",
        r"^say that again",
        r"^i didn'?t (catch|hear|get)\b",
        r"^i (can'?t|couldn'?t) (hear|understand)\b",
        r"^sorry\s*,?\s*(what|i didn)",
        r"^wait\s*,?\s*what",
        r"what was that last",
        r"what did you (just )?say",
    ]

    for pattern in explicit_non_understanding:
        if re.search(pattern, text_lower):
            return "positive", "explicit_non_understanding"

    words = text_lower.split()
    word_count = len(words)

    # Strong negative indicators: understanding confirmations
    understanding_confirmations = {
        "yeah", "yes", "okay", "ok", "right", "ah", "uh-huh", "mm-hmm",
        "mm", "i see", "i understand", "got it", "sure", "true",
        "exactly", "absolutely", "indeed", "of course", "that's right",
        "that's true", "fair enough", "makes sense", "aye",
    }

    if text_clean in understanding_confirmations:
        return "negative", "understanding_confirmation"

    # "Hmm." / "sorry." WITHOUT question mark = acknowledgment/apology, NOT HDM
    if text_clean in {"hmm", "hm", "sorry", "mm"} and not has_question:
        return "negative", "acknowledgment_not_question"

    # Elicit-comment-understanding: speaker checking if OTHERS understand
    # Not hearing difficulty unless it's also a self non-understanding signal
    if da_type == "elicit-comment-understanding":
        is_self_nonunderstanding = any(
            re.search(p, text_lower)
            for p in strong_positive_patterns + explicit_non_understanding
        )
        if not is_self_nonunderstanding:
            return "negative", "elicit_not_self_difficulty"

    # Long utterances = explanations/restatements, not hearing difficulty
    if duration_ms > 3000 or word_count > 10:
        return "negative", "too_long"

    # Short question-word questions WITH '?' — distinguish hearing vs content
    if has_question and word_count <= 5:
        # 1-2 word questions like "What?", "Who?", "A what?" = hearing difficulty
        if word_count <= 3 and any(w in text_clean.split() for w in {"what", "huh", "sorry", "pardon"}):
            return "positive", "short_question"

        # "The what?", "A what?", "It's what?" pattern = hearing difficulty
        if "what" in text_clean.split() and word_count <= 4:
            # Check it's not a semantic question like "What button?"
            # Pattern: [article/pronoun] + "what" = hearing difficulty
            # Pattern: "what" + [noun] = could be semantic
            what_idx = text_clean.split().index("what")
            if what_idx > 0:  # "what" is not the first word → "a what?", "the what?"
                return "positive", "short_question"

        # Content/semantic questions: "How high?", "Where is it?", "Which one?"
        # These are NOT hearing difficulty — speaker heard fine, asking for info
        # Collins et al.: "non-understanding due to semantics...asks for clarification
        # of the meaning or intent" → EXCLUDE
        return "negative", "semantic_question"

    # Restatement patterns (speaker repeating back what they heard)
    if word_count > 5:
        return "negative", "likely_restatement"

    # Short affirmative responses
    if word_count <= 2 and text_clean in {
        "ah", "oh", "mm", "hmm", "uh-huh", "mm-hmm", "mhm", "okay",
        "ok", "yeah", "yep", "right", "sure", "true", "yes", "aye",
        "no", "nah", "nope",
    }:
        return "negative", "short_acknowledgment"

    # Remaining short utterances
    if word_count <= 3:
        return "negative", "short_ambiguous_excluded"

    return "negative", "default_exclude"


def main():
    input_path = Path("data/hdm_annotations.json")
    with open(input_path) as f:
        all_hdm = json.load(f)

    print(f"Total annotations: {len(all_hdm)}")

    classified = {"positive": [], "negative": [], "ambiguous": []}
    reason_counts = Counter()

    for h in all_hdm:
        label, reason = is_hearing_difficulty(h["text"], h["duration_ms"], h["da_type"])
        h["hdm_label"] = label
        h["filter_reason"] = reason
        classified[label].append(h)
        reason_counts[reason] += 1

    print(f"\nClassification results:")
    for label, items in classified.items():
        print(f"  {label}: {len(items)}")

    print(f"\nFilter reasons:")
    for reason, count in reason_counts.most_common():
        print(f"  {reason}: {count}")

    # Show positive examples
    print(f"\n--- Positive examples (hearing difficulty) ---")
    for h in classified["positive"][:20]:
        print(f"  [{h['meeting_id']}.{h['speaker']}] \"{h['text']}\" "
              f"({h['duration_ms']:.0f}ms) [{h['filter_reason']}]")

    # Show ambiguous examples
    print(f"\n--- Ambiguous examples ---")
    for h in classified["ambiguous"][:20]:
        print(f"  [{h['meeting_id']}.{h['speaker']}] \"{h['text']}\" "
              f"({h['duration_ms']:.0f}ms) [{h['filter_reason']}]")

    # Only include positives — no ambiguous. Matches Collins et al.'s strict
    # "signal-non-understanding" criterion: speaker didn't HEAR what was said.
    hdm_positive = classified["positive"]

    print(f"\n=== Final HDM dataset: {len(hdm_positive)} positive instances ===")
    print(f"(from {len(set(h['meeting_id'] for h in hdm_positive))} meetings)")

    # Save filtered results
    with open("data/hdm_filtered.json", "w") as f:
        json.dump(hdm_positive, f, indent=2)

    with open("data/hdm_filtered.csv", "w", newline="") as f:
        if hdm_positive:
            writer = csv.DictWriter(f, fieldnames=hdm_positive[0].keys())
            writer.writeheader()
            writer.writerows(hdm_positive)

    # Save all with labels for reference
    with open("data/hdm_all_classified.json", "w") as f:
        json.dump(all_hdm, f, indent=2)

    print(f"\nSaved filtered HDMs to data/hdm_filtered.json")


if __name__ == "__main__":
    main()
