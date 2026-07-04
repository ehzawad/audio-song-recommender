#!/usr/bin/env python
"""Genre-purity@k evaluation for a built recommendation index.

For a random sample of indexed tracks, checks how often the top-k neighbors
share the query track's genre. Requires metadata.csv (saved alongside every
index) to have a non-null `genre` column, which is true for FMA-derived
indexes.

Usage:
    python scripts/eval_genre_purity.py --artifacts-dir artifacts/fma_small_audio
    python scripts/eval_genre_purity.py --artifacts-dir artifacts/fma_small_clap --k 10 --n 200
"""
from __future__ import annotations

import argparse
import random

import pandas as pd

from music_similarity_rec.recommender import MusicRecommender


def genre_purity_at_k(artifacts_dir: str, k: int, n: int, seed: int) -> tuple[float, int, int]:
    recommender = MusicRecommender.load(artifacts_dir)
    metadata = pd.read_csv(f"{artifacts_dir}/metadata.csv")
    metadata["track_id"] = metadata["track_id"].astype(str)
    metadata = metadata.dropna(subset=["genre"])

    random.seed(seed)
    pool = metadata["track_id"].tolist()
    sample = random.sample(pool, min(n, len(pool)))

    genre_by_id = metadata.set_index("track_id")["genre"]
    hits = total = 0
    for track_id in sample:
        query_genre = genre_by_id[track_id]
        results = recommender.recommend_by_track_id(track_id, k=k)
        hits += (results["genre"] == query_genre).sum()
        total += len(results)
    return hits / total, hits, total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts-dir", required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--n", type=int, default=50, help="number of query tracks to sample")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    purity, hits, total = genre_purity_at_k(args.artifacts_dir, args.k, args.n, args.seed)
    print(f"genre purity@{args.k} over {args.n} queries: {purity:.3f} ({hits}/{total})")


if __name__ == "__main__":
    main()
