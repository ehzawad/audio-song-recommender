#!/usr/bin/env bash
set -euo pipefail

# Start server in another terminal:
# music-rec serve --artifacts-dir artifacts/toy --host 127.0.0.1 --port 8000

curl 'http://127.0.0.1:8000/health'

curl 'http://127.0.0.1:8000/tracks/a_440/similar?k=3'

curl -X POST 'http://127.0.0.1:8000/audio/similar?k=3' \
  -F 'file=@data/toy_audio/a_445.wav'
