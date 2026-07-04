"""CLAP embed fma_small with laion/clap-htsat-unfused on GPU 0 (A5000).

v2: mel extraction runs inside the decode workers (parallel), the main
process only batches mels through the GPU. Checkpoints every 500 tracks so
the run survives restarts; already-embedded tracks are skipped on resume.
"""
from __future__ import annotations

import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DECODE_WORKERS = 20
GPU_BATCH = 48  # windows per GPU call (16 tracks x 3 windows)
CLAP_SR = 48_000
WINDOW = 10.0
MODEL_ID = "laion/clap-htsat-unfused"

AUDIO_DIR = Path("data/raw/fma_small")
ARTIFACTS_DIR = Path("artifacts/fma_small_clap2")
METADATA_CSV = "artifacts/fma_small_audio/metadata.csv"
CHECKPOINT = ARTIFACTS_DIR / "checkpoint.npz"

_fe = None


def _init() -> None:
    global _fe
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    from transformers import ClapFeatureExtractor
    _fe = ClapFeatureExtractor.from_pretrained(MODEL_ID)


def decode_mel(job: tuple[str, str]):
    """Load 30s, cut three 10s windows, return mel features (n,1001,64)."""
    path, track_id = job
    try:
        import librosa
        y, _ = librosa.load(path, sr=CLAP_SR, mono=True, duration=3 * WINDOW)
        if y.size == 0:
            raise ValueError("empty audio")
        win = int(WINDOW * CLAP_SR)
        chunks = [y[s : s + win] for s in range(0, max(len(y) - win // 2, 1), win)][:3]
        chunks = [np.pad(c, (0, win - len(c))) if len(c) < win else c for c in chunks]
        feats = _fe(chunks, sampling_rate=CLAP_SR, return_tensors="np")["input_features"]
        return track_id, feats.astype(np.float32), None
    except Exception as exc:
        return track_id, None, repr(exc)


def save_checkpoint(ids: list[str], embs: list[np.ndarray]) -> None:
    np.savez(CHECKPOINT, ids=np.array(ids), embs=np.vstack(embs) if embs else np.zeros((0, 512)))


def main() -> None:
    import torch
    from transformers import ClapModel

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    ids: list[str] = []
    embs: list[np.ndarray] = []
    if CHECKPOINT.exists():
        ck = np.load(CHECKPOINT, allow_pickle=False)
        ids = [str(x) for x in ck["ids"]]
        embs = [row[None, :] for row in ck["embs"]]
        print(f"resumed checkpoint with {len(ids)} tracks", flush=True)

    device = "cuda:0"
    model = ClapModel.from_pretrained(MODEL_ID).to(device).eval()
    print(f"model loaded on {torch.cuda.get_device_name(0)}", flush=True)

    files = sorted(AUDIO_DIR.rglob("*.mp3"))
    done_ids = set(ids)
    # known-corrupt fma_small files (also failed in the handcrafted-feature run);
    # audioread's fallback can hang on them, so skip outright
    corrupt = {"99134", "108925", "133297"}
    jobs = []
    for p in files:
        tid = p.stem.lstrip("0") or "0"
        if tid not in done_ids and tid not in corrupt:
            jobs.append((str(p), tid))
    print(f"embedding {len(jobs)} remaining files: {DECODE_WORKERS} workers", flush=True)

    errors: list[dict] = []
    buf_ids: list[str] = []
    buf_mel: list[np.ndarray] = []
    buf_counts: list[int] = []

    def flush() -> None:
        if not buf_mel:
            return
        x = torch.from_numpy(np.concatenate(buf_mel, axis=0)).to(device)
        with torch.no_grad():
            e = model.get_audio_features(input_features=x)
            if not torch.is_tensor(e):
                e = e.pooler_output
        e = torch.nn.functional.normalize(e, dim=-1).cpu().numpy()
        pos = 0
        for tid, n in zip(buf_ids, buf_counts, strict=True):
            v = e[pos : pos + n].mean(axis=0)
            embs.append((v / (np.linalg.norm(v) + 1e-12))[None, :])
            ids.append(tid)
            pos += n
        buf_ids.clear(); buf_mel.clear(); buf_counts.clear()

    done = 0
    ctx = mp.get_context("spawn")  # fork after CUDA init deadlocks workers
    with ctx.Pool(DECODE_WORKERS, initializer=_init) as pool:
        for track_id, mel, err in pool.imap_unordered(decode_mel, jobs, chunksize=2):
            done += 1
            if err is not None:
                errors.append({"track_id": track_id, "error": err})
            else:
                buf_ids.append(track_id)
                buf_mel.append(mel)
                buf_counts.append(mel.shape[0])
                if sum(buf_counts) >= GPU_BATCH:
                    flush()
            if done % 250 == 0:
                flush()
                save_checkpoint(ids, embs)
                print(f"  {done}/{len(jobs)} done ({len(errors)} errors) [checkpointed]", flush=True)
    flush()
    save_checkpoint(ids, embs)

    print(f"embedding finished: {len(ids)} total ok, {len(errors)} failed this run", flush=True)
    if not ids:
        sys.exit("no embeddings")

    mat = np.vstack(embs)
    features = pd.DataFrame(
        mat, index=pd.Index(ids, name="track_id"),
        columns=[f"clap_{i:03d}" for i in range(mat.shape[1])],
    ).sort_index()

    from music_similarity_rec.index import IndexConfig, SimilarityIndex
    metadata = pd.read_csv(METADATA_CSV)
    if errors:
        pd.DataFrame(errors).to_csv(ARTIFACTS_DIR / "embed_errors.csv", index=False)
    index = SimilarityIndex.build(features, config=IndexConfig())
    out = index.save(ARTIFACTS_DIR, features=features, metadata=metadata)
    print(f"Saved CLAP index to {out}", flush=True)


if __name__ == "__main__":
    main()
