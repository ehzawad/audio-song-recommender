from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from music_similarity_rec.api import create_app
from music_similarity_rec.config import load_config
from music_similarity_rec.datasets import FMA_PACKAGES, download_fma_package
from music_similarity_rec.features import AudioFeatureExtractor, FeatureConfig
from music_similarity_rec.fma import (
    enrich_fma_paths,
    filter_fma_metadata,
    load_fma_feature_dataset,
    load_fma_tracks,
)
from music_similarity_rec.index import IndexConfig, SimilarityIndex
from music_similarity_rec.metadata import build_local_metadata, merge_metadata
from music_similarity_rec.paths import ensure_dir
from music_similarity_rec.recommender import MusicRecommender
from music_similarity_rec.toydata import create_toy_audio


def _print_table(df: pd.DataFrame, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(df.where(pd.notna(df), None).to_dict(orient="records"), indent=2))
    else:
        with pd.option_context("display.max_colwidth", 60, "display.width", 140):
            print(df.to_string(index=False))


def cmd_download_fma(args: argparse.Namespace) -> None:
    for package in args.package:
        zip_path = download_fma_package(
            package,
            out_dir=args.out_dir,
            unpack=not args.no_unpack,
            overwrite=args.overwrite,
        )
        print(f"OK: {zip_path}")


def cmd_make_toy_data(args: argparse.Namespace) -> None:
    out = create_toy_audio(args.out_dir, seconds=args.seconds)
    print(f"Created toy audio corpus in {out}")


def cmd_build_audio_index(args: argparse.Namespace) -> None:
    app_config = load_config(args.config)
    feature_config = FeatureConfig.from_settings(app_config.feature_extraction)
    index_config = IndexConfig.from_settings(app_config.index)

    id_mode = args.id_mode
    if id_mode == "auto":
        id_mode = "fma" if args.fma_tracks_csv else "relative"
    metadata, files, track_ids = build_local_metadata(args.audio_dir, id_mode=id_mode)

    if args.fma_tracks_csv:
        fma_meta = load_fma_tracks(args.fma_tracks_csv)
        if args.fma_subset or args.fma_split:
            fma_meta = filter_fma_metadata(fma_meta, subset=args.fma_subset, split=args.fma_split)
        metadata = merge_metadata(metadata, fma_meta)

    if args.limit is not None:
        metadata = metadata.head(args.limit).copy()
        keep_ids = set(metadata["track_id"].astype(str))
        pairs = [(path, tid) for path, tid in zip(files, track_ids, strict=True) if tid in keep_ids]
        files = [path for path, _ in pairs]
        track_ids = [tid for _, tid in pairs]

    extractor = AudioFeatureExtractor(feature_config)
    features, errors = extractor.extract_many(files, track_ids, show_progress=not args.no_progress)
    if not errors.empty:
        error_path = ensure_dir(args.artifacts_dir) / "feature_errors.csv"
        errors.to_csv(error_path, index=False)
        print(f"Feature extraction skipped {len(errors)} files; see {error_path}")

    index = SimilarityIndex.build(features, config=index_config)
    out = index.save(args.artifacts_dir, features=features, metadata=metadata)
    print(f"Saved audio-based recommendation index to {out}")


def cmd_build_fma_feature_index(args: argparse.Namespace) -> None:
    app_config = load_config(args.config)
    index_config = IndexConfig.from_settings(app_config.index)
    features, metadata = load_fma_feature_dataset(
        tracks_csv=args.tracks_csv,
        features_csv=args.features_csv,
        subset=args.subset,
        split=args.split,
        limit=args.limit,
    )
    if args.audio_dir:
        metadata = enrich_fma_paths(metadata, args.audio_dir)
    index = SimilarityIndex.build(features, config=index_config)
    out = index.save(args.artifacts_dir, features=features, metadata=metadata)
    print(f"Saved FMA precomputed-feature index to {out}")
    print("Note: this index is ideal for recommend-track. For recommend-audio, build with build-audio-index so query and corpus features match exactly.")


def cmd_recommend_track(args: argparse.Namespace) -> None:
    recommender = MusicRecommender.load(args.artifacts_dir)
    df = recommender.recommend_by_track_id(args.track_id, k=args.k)
    _print_table(df, as_json=args.json)


def cmd_recommend_audio(args: argparse.Namespace) -> None:
    recommender = MusicRecommender.load(args.artifacts_dir)
    df = recommender.recommend_by_audio_file(args.query_audio, k=args.k)
    _print_table(df, as_json=args.json)


def cmd_serve(args: argparse.Namespace) -> None:
    import uvicorn

    app = create_app(args.artifacts_dir)
    uvicorn.run(app, host=args.host, port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="music-rec",
        description="Audio-based similar-song recommender: build indexes, query by track/audio, or serve an API.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download-fma", help="Download and optionally unpack FMA packages.")
    p.add_argument("package", nargs="+", choices=sorted(FMA_PACKAGES), help="FMA package(s) to download.")
    p.add_argument("--out-dir", default="data/raw")
    p.add_argument("--no-unpack", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_download_fma)

    p = sub.add_parser("make-toy-data", help="Create a tiny synthetic audio corpus for smoke tests.")
    p.add_argument("--out-dir", default="data/toy_audio")
    p.add_argument("--seconds", type=float, default=4.0)
    p.set_defaults(func=cmd_make_toy_data)

    p = sub.add_parser("build-audio-index", help="Extract audio features from a local folder and build a nearest-neighbor index.")
    p.add_argument("--audio-dir", required=True)
    p.add_argument("--artifacts-dir", default="artifacts/local")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-progress", action="store_true")
    p.add_argument(
        "--id-mode",
        choices=["auto", "relative", "stem", "fma"],
        default="auto",
        help="How track IDs are derived from audio paths. auto uses fma when --fma-tracks-csv is supplied; otherwise relative.",
    )
    p.add_argument("--fma-tracks-csv", default=None, help="Optional FMA tracks.csv to enrich metadata.")
    p.add_argument("--fma-subset", default=None, help="Optional FMA subset filter, e.g. small.")
    p.add_argument("--fma-split", default=None, help="Optional FMA split filter, e.g. training.")
    p.set_defaults(func=cmd_build_audio_index)

    p = sub.add_parser("build-fma-feature-index", help="Build a fast index from FMA metadata/features CSV files.")
    p.add_argument("--tracks-csv", required=True)
    p.add_argument("--features-csv", required=True)
    p.add_argument("--artifacts-dir", default="artifacts/fma_features")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--subset", default="small")
    p.add_argument("--split", default=None)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--audio-dir", default=None, help="Optional local FMA audio directory to attach paths.")
    p.set_defaults(func=cmd_build_fma_feature_index)

    p = sub.add_parser("recommend-track", help="Recommend similar songs by an indexed track ID.")
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--track-id", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recommend_track)

    p = sub.add_parser("recommend-audio", help="Recommend similar indexed songs for an external query audio file.")
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--query-audio", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_recommend_audio)

    p = sub.add_parser("serve", help="Serve a small FastAPI app over saved recommender artifacts.")
    p.add_argument("--artifacts-dir", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
