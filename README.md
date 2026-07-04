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
- Optional: a neural embedding upgrade path (`scripts/clap_embed.py`) using a pretrained CLAP model, plus a reproducible genre-purity evaluation script.

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

## Neural embeddings (optional upgrade)

The default `AudioFeatureExtractor` produces 309-dim hand-engineered features (MFCC, chroma, spectral shape, RMS, ZCR, tempo). `scripts/clap_embed.py` swaps this for [CLAP](https://huggingface.co/laion/clap-htsat-unfused) (`laion/clap-htsat-unfused`), a pretrained contrastive audio-text model, while keeping the same `SimilarityIndex`/CLI/API downstream.

```zsh
python -m pip install torch transformers
python scripts/clap_embed.py --limit 100   # smoke test
python scripts/clap_embed.py               # full fma_small (8000 tracks)
```

Each track is embedded as three deterministic 10-second windows, L2-normalized and averaged. The script checkpoints every 250 tracks to `artifacts/fma_small_clap2/checkpoint.npz`, so a run can be safely interrupted and resumed. It uses `CUDA_VISIBLE_DEVICES` if set; without a GPU it falls back to CPU (slower).

**Known checkpoint issue:** `laion/larger_clap_music` produces collapsed/degenerate audio embeddings through the `transformers` `ClapModel` API (verified on both transformers 4.57 and 5.13) — it scores *worse* than the classical baseline and should not be used. `laion/clap-htsat-unfused` does not have this problem; it is the checkpoint `clap_embed.py` uses by default.

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

scripts/
  run_toy_demo.sh                  toy pipeline demo
  run_fma_small_audio.sh            fully audio-based FMA small demo
  run_fma_small_fast_features.sh    precomputed-features FMA small demo
  clap_embed.py                     neural (CLAP) embedding pipeline
  eval_genre_purity.py              genre-purity@k evaluation
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

## Evaluation

### Genre purity@k

Since FMA has no ground-truth "these songs sound alike" labels, genre agreement among top-k neighbors is a standard proxy: a system with no acoustic signal would match genre roughly `1 / num_genres` of the time (0.125 for FMA's 8 genres); a good content-based system should score well above that.

Reproduce with:

```zsh
python scripts/eval_genre_purity.py --artifacts-dir artifacts/fma_small_audio --k 10 --n 50
```

Measured on `fma_small` (8,000 tracks, 8 genres), `k=10`, 50 sampled query tracks, seed 0:

| Index | Feature space | Genre purity@10 |
|---|---|---|
| `fma_small_features` | FMA precomputed, 518-dim | 0.452 |
| `fma_small_audio` | This project's `AudioFeatureExtractor`, 309-dim | 0.464 |
| `fma_small_clap` (CLAP, see below) | `laion/clap-htsat-unfused`, 512-dim | 0.600 |
| — | random baseline | 0.125 |

All three indexes clear the random baseline by 3.6–4.8x; the pretrained CLAP embeddings noticeably outperform both hand-engineered feature sets on this proxy metric. Numbers will shift somewhat with a different sample size/seed or FMA subset — treat them as directional, not exact.

### Robustness to new/degraded audio

`recommend-audio` and `POST /audio/similar` exist specifically so a query song never has to be one already in the index. To check this actually generalizes (not just "the model memorized the corpus"), query with audio the index has not seen in its original form:

- **Mildly degraded** (re-encoded to 64 kbps MP3, volume-shifted): the CLAP index retrieved the original track in its top-3 for 3 of 4 test clips (2 at rank 1).
- **Heavily degraded** (downsampled to 8 kHz, band-limited to telephone bandwidth 300–3400 Hz, added noise): self-retrieval largely fails for both feature spaces, but the CLAP index still frequently ranks same-genre tracks highest — the embedding degrades gracefully rather than catastrophically.

Takeaway: the system is robust to realistic quality variation (different encodes/bitrates of the same recording) but is not an audio fingerprinting system — it will not identify a track through severe telephone-grade filtering. For exact-match "what song is this" identification under heavy distortion, pair this with a dedicated fingerprinting approach (e.g., chromaprint/Shazam-style hashing) rather than relying on similarity embeddings alone.

### Other proxy tasks worth adding

- Artist leakage checks: whether the system over-recommends the same artist.
- A/B listening tests: ask listeners whether the recommendations make acoustic sense.
- Query-excerpt stability: encode different excerpts of the *same* track and verify the neighborhoods stay similar.

## Known limitations

This is a content-based baseline. It does not learn from user behavior, playlists, skips, saves, or session context. That is why it will not fully reproduce Spotify’s recommender, which is a hybrid system. The upside is that this project is inspectable: every recommendation comes from feature-space distance.

For higher-quality audio embeddings, see [Neural embeddings](#neural-embeddings-optional-upgrade) above — `scripts/clap_embed.py` swaps `AudioFeatureExtractor` for a pretrained CLAP model while keeping the rest of the system (index, CLI, API) unchanged: vectors go in, nearest-neighbor retrieval comes out.

`recommend-audio`/`POST /audio/similar` only work correctly against an index built with a matching feature extractor: query and corpus vectors must live in the same feature space (see [Robustness to new/degraded audio](#robustness-to-newdegraded-audio)).

## Licensing notes

The code in this generated project is MIT licensed. FMA metadata and audio have their own licenses. The FMA authors note that the metadata is CC BY 4.0, the audio copyright remains with artists and is distributed under artist-selected licenses, and the dataset is meant for research use. Check track-level licenses before redistributing audio or building a commercial product.
