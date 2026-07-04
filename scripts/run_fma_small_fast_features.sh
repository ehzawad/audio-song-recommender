#!/usr/bin/env bash
set -euo pipefail

# This path is faster because it uses FMA's precomputed features.csv.
# Use this for recommend-track. For recommend-audio, use run_fma_small_audio.sh.
python -m music_similarity_rec.cli download-fma metadata --out-dir data/raw
python -m music_similarity_rec.cli build-fma-feature-index \
  --tracks-csv data/raw/fma_metadata/tracks.csv \
  --features-csv data/raw/fma_metadata/features.csv \
  --subset small \
  --artifacts-dir artifacts/fma_small_features
python -m music_similarity_rec.cli recommend-track \
  --artifacts-dir artifacts/fma_small_features \
  --track-id 2 \
  --k 10
