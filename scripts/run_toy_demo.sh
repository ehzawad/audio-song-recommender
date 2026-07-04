#!/usr/bin/env bash
set -euo pipefail

python -m music_similarity_rec.cli make-toy-data --out-dir data/toy_audio
python -m music_similarity_rec.cli build-audio-index \
  --audio-dir data/toy_audio \
  --artifacts-dir artifacts/toy
python -m music_similarity_rec.cli recommend-track \
  --artifacts-dir artifacts/toy \
  --track-id a_440 \
  --k 3
python -m music_similarity_rec.cli recommend-audio \
  --artifacts-dir artifacts/toy \
  --query-audio data/toy_audio/a_445.wav \
  --k 3
