# Data Leakage Audit & Validation Report

This document describes the validation process used to confirm that the GPT-4o Audio v4 classifier results (F1 = 0.94) are free from data leakage and methodologically sound.

## Overview

The classifier uses **Monte Carlo cross-validation** with 5 splits across 38 AMI meetings. To ensure no information from the test set leaks into training or inference, we performed an 8-point audit covering split integrity, few-shot selection, ground truth consistency, temporal isolation, and transcript sourcing.

## Audit Results

| # | Check | Status | Details |
|---|-------|--------|---------|
| 1 | Meeting-level split integrity | PASS | All 5 splits have 0 meetings appearing in both train and test. 30 train / 8 test per split, all 38 meetings assigned. |
| 2 | Example-level assignment | PASS | Every example maps to exactly one partition (train or test) via its meeting ID. 0 overlapping examples, 0 unassigned. |
| 3 | Few-shot selection from training only | PASS | `select_shot_examples_v4()` filters candidates by `train_indices` before selection. All 20 shots (12P + 8N) come from training meetings in every split. |
| 4 | Human labels do not alter ground truth | PASS | Human labels (`hdm_labels.json`) are used only to select high-quality few-shot examples. Ground truth labels remain the original AMI annotations. |
| 5 | Predictions align with test set sizes | PASS | Prediction counts match expected test set sizes for all splits. `true_labels` in results match `labels.npy` for corresponding test indices. |
| 6 | Temporal isolation (4s segments) | PASS | 0 temporal overlaps between positive and negative segments within any meeting. |
| 7 | Extended context (12s) isolation | PASS | 23 extended clips share some audio within the same meeting, but meeting-level splitting guarantees 0 cross-partition overlaps. |
| 8 | Transcript source | PASS | All transcripts come from the AMI corpus annotations, not from model output. Positives have transcripts (105/105); negatives do not (0/1050, as expected). |

**Verdict: No data leakage detected.**

## Detailed Methodology

### 1. Meeting-Level Split Integrity

Splits are created at the **conversation level** (`build_dataset.py:154-170`). The 38 meetings are shuffled with a fixed seed and divided 80/20. A meeting never appears in both train and test for the same split.

```
Split 1: train=30, test=8, overlap=0, unassigned=0
Split 2: train=30, test=8, overlap=0, unassigned=0
Split 3: train=30, test=8, overlap=0, unassigned=0
Split 4: train=30, test=8, overlap=0, unassigned=0
Split 5: train=30, test=8, overlap=0, unassigned=0
```

### 2. Example-Level Assignment

Each of the 1,155 examples (105 positive + 1,050 negative) is assigned to train or test based solely on its `meeting_id`. No example appears in both partitions.

```
Split 1: train=990, test=165, overlap=0, unassigned=0
Split 2: train=946, test=209, overlap=0, unassigned=0
Split 3: train=891, test=264, overlap=0, unassigned=0
Split 4: train=869, test=286, overlap=0, unassigned=0
Split 5: train=924, test=231, overlap=0, unassigned=0
```

### 3. Few-Shot Selection

The v4 classifier uses 20-shot prompting (12 positive + 8 negative). Shot candidates are filtered by `train_indices` before any selection occurs (`gpt4o_audio_classifier_v4.py`):

1. Only examples whose `meeting_id` is in the training set are eligible
2. Human-verified positives (`hdm_labels.json: "yes"`) are preferred for positive shots
3. Human-rejected positives (`hdm_labels.json: "no"`) serve as hard negatives
4. Remaining negative shots are drawn from the training set's random negatives

No few-shot example from any split came from a test meeting.

### 4. Human Labels vs Ground Truth

84 of the 105 positive examples were manually reviewed. 7 were relabeled as "no" (not true HDMs). However:

- **Ground truth is unchanged**: evaluation uses the original AMI annotations (`labels.npy`)
- **Human labels are used only for shot selection**: to pick better few-shot examples
- **This makes evaluation conservative**: if the human reviewer is correct, some model "false negatives" are actually correct rejections, meaning true F1 could be *higher*

Relabeled indices: `[1, 43, 60, 61, 74, 76, 78]` — all still carry ground truth label = 1.

### 5. Prediction-Metadata Alignment

For every split, we verified:
- `len(predictions)` matches the expected number of test examples
- `true_labels` in the results file matches the corresponding entries in `labels.npy`

```
Split 1: expected=165, preds=165, labels_match=True
Split 2: expected=209, preds=209, labels_match=True
Split 3: expected=264, preds=264, labels_match=True
Split 4: expected=286, preds=286, labels_match=True
Split 5: expected=231, preds=231, labels_match=True
```

### 6. Temporal Isolation (4s Segments)

No 4-second positive segment overlaps temporally with any 4-second negative segment within the same meeting. The `build_dataset.py` negative sampler explicitly avoids HDM intervals (expanded by the segment duration) when selecting random timesteps.

### 7. Extended Context (12s) Temporal Overlap

The v4 classifier sends 12 seconds of audio per inference (4s context + 4s segment + 4s after). Within the same meeting, 23 extended clips share some audio content with clips of the opposite class. This is **not a leakage concern** because:

- All clips from the same meeting go to the **same partition** (train or test)
- Cross-partition overlap is **0** for all 5 splits (guaranteed by meeting-level splitting)

### 8. Transcript Source

Transcripts are from the **AMI corpus annotations** embedded in `dataset_meta.json` at build time. They are not generated by the model at inference. All 105 positive examples have transcripts; all 1,050 negative examples have empty transcripts (they use the full meeting mix audio with no specific speaker).

## Reproducibility

### Repeated Runs

The v4 classifier was run 3 times to verify consistency:

| Run | Avg F1 | Std | Per-split F1 |
|-----|--------|-----|-------------|
| 1 | 0.9434 | 0.0487 | [0.89, 1.00, 0.93, 1.00, 0.89] |
| 2 | 0.9428 | 0.0285 | [0.93, 0.97, 0.91, 0.98, 0.92] |
| 3 | 0.9442 | 0.0468 | [0.89, 1.00, 0.91, 1.00, 0.92] |

**Cross-run mean F1: 0.9435**. Variance between runs is minimal, confirming stable performance.

### Fixed Seed

- Dataset construction uses `RANDOM_SEED = 42` for reproducible splits and negative sampling
- Few-shot selection uses `seed = 42 + split_idx` per split

## Minor Notes

- The v4 classifier hardcodes `n_positives = 149`, but only 105 positives exist. Indices 105-148 are never present in `train_indices`, so this has no effect on shot selection or evaluation.
- The validation dashboard (`src/validation_dashboard.py`) allows manual inspection of every test prediction with audio playback, enabling human verification of TP/FP/FN/TN classifications.

## How to Reproduce This Audit

```bash
# Run the leakage check script
python src/validation_audit.py

# Launch the validation dashboard for manual inspection
python src/validation_dashboard.py
# Then open http://localhost:8766
```
