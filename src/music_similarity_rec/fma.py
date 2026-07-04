from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def _flatten_col(col: object) -> str:
    if not isinstance(col, tuple):
        return str(col).strip().lower().replace(" ", "_")
    parts: list[str] = []
    for part in col:
        text = str(part).strip()
        if not text or text.lower().startswith("unnamed"):
            continue
        parts.append(text.lower().replace(" ", "_"))
    return "__".join(parts)


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [_flatten_col(col) for col in out.columns]
    return out


def load_fma_features(features_csv: str | Path) -> pd.DataFrame:
    """Load FMA `features.csv` and flatten its multi-row header."""
    features = pd.read_csv(features_csv, index_col=0, header=[0, 1, 2])
    features = _flatten_columns(features)
    features.index = features.index.astype(str)
    features.index.name = "track_id"
    return features.apply(pd.to_numeric, errors="coerce")


def load_fma_tracks(tracks_csv: str | Path) -> pd.DataFrame:
    """Load FMA `tracks.csv` and return normalized recommendation metadata."""
    tracks = pd.read_csv(tracks_csv, index_col=0, header=[0, 1])
    flat = _flatten_columns(tracks)
    flat.index = flat.index.astype(str)
    flat.index.name = "track_id"

    def pick(*names: str) -> pd.Series:
        for name in names:
            if name in flat.columns:
                return flat[name]
        return pd.Series([None] * len(flat), index=flat.index)

    metadata = pd.DataFrame(
        {
            "track_id": flat.index.astype(str),
            "title": pick("track__title", "title"),
            "artist": pick("artist__name", "artist_name"),
            "album": pick("album__title", "album_title"),
            "genre": pick("track__genre_top", "genre_top"),
            "split": pick("set__split", "split"),
            "subset": pick("set__subset", "subset"),
        }
    )
    return metadata


def fma_audio_path(audio_dir: str | Path, track_id: str | int) -> Path:
    tid = int(track_id)
    tid_str = f"{tid:06d}"
    return Path(audio_dir).expanduser().resolve() / tid_str[:3] / f"{tid_str}.mp3"


def enrich_fma_paths(metadata: pd.DataFrame, audio_dir: str | Path) -> pd.DataFrame:
    out = metadata.copy()
    out["path"] = [str(fma_audio_path(audio_dir, tid)) for tid in out["track_id"]]
    return out


def filter_fma_metadata(
    metadata: pd.DataFrame,
    subset: str | None = None,
    split: str | None = None,
    track_ids: Iterable[str] | None = None,
) -> pd.DataFrame:
    out = metadata.copy()
    out["track_id"] = out["track_id"].astype(str)
    if subset:
        out = out[out["subset"].astype(str) == subset]
    if split:
        out = out[out["split"].astype(str) == split]
    if track_ids is not None:
        ids = {str(x) for x in track_ids}
        out = out[out["track_id"].isin(ids)]
    return out.reset_index(drop=True)


def load_fma_feature_dataset(
    tracks_csv: str | Path,
    features_csv: str | Path,
    subset: str | None = "small",
    split: str | None = None,
    limit: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = filter_fma_metadata(load_fma_tracks(tracks_csv), subset=subset, split=split)
    features = load_fma_features(features_csv)
    ids = metadata["track_id"].astype(str).tolist()
    features = features.loc[features.index.intersection(ids)].copy()
    metadata = filter_fma_metadata(metadata, track_ids=features.index)
    if limit is not None:
        metadata = metadata.head(limit).copy()
        features = features.loc[metadata["track_id"].astype(str)].copy()
    if features.empty:
        raise ValueError("No FMA feature rows matched the requested filters")
    return features, metadata
