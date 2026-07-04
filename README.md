# Audio Song Recommender

This is an end-to-end, modular starter project for content-based music recommendation: given one song, return songs that sound similar. It is intentionally close to the “Spotify similar songs” idea, but uses only audio/content similarity rather than collaborative filtering, playlists, user histories, location, recency, skips, or editorial signals.

The default dataset target is the **Free Music Archive (FMA)** because it includes MP3 audio, metadata, and precomputed audio features. The same code also works with any local folder of audio files.

## What this project implements

- Dataset download helpers for FMA metadata/audio.
- Audio feature extraction from local `.mp3`, `.wav`, `.flac`, `.ogg`, `.m4a`, or `.aac` files.
- A cosine nearest-neighbor index over standardized audio features.
- Recommendation by indexed track ID.
- Recommendation by uploading/providing a new query audio file.
- A small FastAPI service for HTTP recommendations.
- A tiny generated toy audio corpus so you can smoke-test the whole pipeline without downloading gigabytes.

## Mental model

The pipeline is:

```text
audio files → feature vectors → impute/standardize → nearest-neighbor index → similar songs
```

For a production Spotify-like system, you would usually combine this content model with collaborative filtering, sequence models, editorial/business rules, and feedback loops. Here, the system is deliberately audio-first: it recommends tracks whose acoustic feature vectors are close to the query vector.

## Dataset options

### Recommended: FMA small

FMA small contains 8,000 thirty-second tracks across eight balanced top-level genres. It is large enough to feel real and small enough to run locally. The audio zip is still several GB, so it is not included in this project zip.

### Fast mode: FMA precomputed features

FMA also distributes `features.csv`. This lets you build an index quickly without re-extracting audio. Use this when you want `recommend-track` over FMA tracks.

### Fully audio-based mode

Use `build-audio-index` over FMA audio or any local audio folder. This extracts features using this project’s `AudioFeatureExtractor`, which means `recommend-audio` can compare an external query song in the same feature space.

## Setup

Use Python 3.10+.

```zsh
cd audio-song-recommender
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'
```

For MP3/M4A decoding, installing FFmpeg is recommended:

```zsh
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get update && sudo apt-get install -y ffmpeg
```

## Smoke-test without downloading a dataset

```zsh
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
```

Equivalent script:

```zsh
./scripts/run_toy_demo.sh
```

## Use FMA small, fully audio-based

This is the best match for “based off of audio, suggest similar songs.”

```zsh
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
```

Equivalent script:

```zsh
./scripts/run_fma_small_audio.sh
```

Notes:

- FMA audio files are stored as paths like `data/raw/fma_small/002/002680.mp3`.
- This mode may take a while because it decodes audio and extracts features from every file.
- In `build-audio-index`, `--id-mode auto` uses numeric FMA IDs when `--fma-tracks-csv` is supplied, so a file like `000/000002.mp3` becomes track ID `2`. Without FMA metadata, local IDs are derived from relative file paths.

## Use FMA precomputed features, fast mode

This builds a recommender quickly from `features.csv`. It is excellent for querying “songs similar to this indexed FMA track,” but it should not be used for external query audio unless you generate query vectors with exactly the same feature schema.

```zsh
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
```

Equivalent script:

```zsh
./scripts/run_fma_small_fast_features.sh
```

## Run the API

```zsh
python -m music_similarity_rec.cli serve \
  --artifacts-dir artifacts/toy \
  --host 127.0.0.1 \
  --port 8000
```

Then query it:

```zsh
curl 'http://127.0.0.1:8000/health'

curl 'http://127.0.0.1:8000/tracks/a_440/similar?k=3'

curl -X POST 'http://127.0.0.1:8000/audio/similar?k=3' \
  -F 'file=@data/toy_audio/a_445.wav'
```

## Project layout

```text
src/music_similarity_rec/
  api.py          FastAPI app factory
  cli.py          CLI commands
  config.py       YAML config loader
  datasets.py     FMA download + checksum helpers
  features.py     audio feature extraction
  fma.py          FMA CSV loading/parsing helpers
  index.py        preprocessing + nearest-neighbor index
  metadata.py     local/FMA metadata assembly
  paths.py        filesystem helpers
  recommender.py  high-level recommendation API
  schemas.py      dataclasses and ID helpers
  toydata.py      synthetic audio generator
```

## Configuration

Edit `configs/default.yaml`:

```yaml
feature_extraction:
  sample_rate: 22050
  duration: 30.0
  offset: 0.0
  n_fft: 2048
  hop_length: 512
  n_mfcc: 20
  include_tempo: true

index:
  metric: cosine
  algorithm: brute
  pca_components: null
  random_state: 42
```

Use PCA for very wide feature vectors or if you want a compact retrieval space:

```yaml
index:
  metric: cosine
  algorithm: brute
  pca_components: 64
```

## Tests

```zsh
pytest
```

## Evaluation ideas

Content similarity can be evaluated with several proxy tasks:

- Nearest-neighbor genre purity: how often top-k neighbors share `genre`.
- Artist leakage checks: whether the system recommends the same artist too often.
- A/B listening tests: ask listeners whether the recommendations make acoustic sense.
- Query perturbation: encode different excerpts of the same track and verify stable neighborhoods.

A minimal genre-purity loop is easy to add after recommendations:

```python
same_genre_at_k = (results["genre"] == query_genre).mean()
```

## Known limitations

This is a content-based baseline. It does not learn from user behavior, playlists, skips, saves, or session context. That is why it will not fully reproduce Spotify’s recommender, which is a hybrid system. The upside is that this project is inspectable: every recommendation comes from feature-space distance.

For higher-quality audio embeddings, replace `AudioFeatureExtractor` with a pretrained music embedding model and keep the rest of the system nearly unchanged: vectors go in, nearest-neighbor retrieval comes out.

## Licensing notes

The code in this generated project is MIT licensed. FMA metadata and audio have their own licenses. The FMA authors note that the metadata is CC BY 4.0, the audio copyright remains with artists and is distributed under artist-selected licenses, and the dataset is meant for research use. Check track-level licenses before redistributing audio or building a commercial product.
