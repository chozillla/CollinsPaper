#!/bin/bash
# Run the full replication pipeline
# Usage: bash run_pipeline.sh

set -e

echo "============================================"
echo "Collins et al. Replication on AMI Corpus"
echo "============================================"

echo ""
echo "Step 1: Parse AMI annotations"
uv run python src/parse_ami_annotations.py

echo ""
echo "Step 2: Filter for hearing difficulty moments"
uv run python src/filter_hdm.py

echo ""
echo "Step 3: Download audio (if needed)"
uv run python src/download_audio.py

echo ""
echo "Step 4: Build dataset"
uv run python src/build_dataset.py

echo ""
echo "Step 5: Run ASR Hotword Baseline"
uv run python src/baseline_hotword.py

echo ""
echo "Step 6: Run Wav2Vec 2.0 Classifier"
uv run python src/wav2vec_classifier.py

echo ""
echo "Step 7: Run Audio LM Prompting"
uv run python src/audio_lm_prompting.py

echo ""
echo "Step 8: Generate comparison results"
uv run python src/evaluate_all.py

echo ""
echo "============================================"
echo "Pipeline complete! Results in results/"
echo "============================================"
