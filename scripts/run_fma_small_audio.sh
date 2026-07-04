#!/usr/bin/env bash
set -euo pipefail

# This is the fully audio-based path. It downloads FMA metadata + fma_small audio,
# extracts features from the MP3 files, builds a cosine nearest-neighbor index,
# then queries similar tracks.
python -m music_similarity_rec.cli download-fma metadata small --out-dir data/raw
python -m music_similarity_rec.cli build-audio-index \
  --audio-dir data/raw/fma_small \
  --fma-tracks-csv data/raw/fma_metadata/tracks.csv \
  --fma-subset small \
  --artifacts-dir artifacts/fma_small_audio
python -m music_similarity_rec.cli recommend-track \
  --artifacts-dir artifacts/fma_small_audio \
  --track-id 2 \
  --k 10
