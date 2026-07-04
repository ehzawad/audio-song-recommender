from __future__ import annotations

from pathlib import Path

import pandas as pd

from music_similarity_rec.paths import iter_audio_files
from music_similarity_rec.schemas import TrackRecord, local_track_id


def _fma_id_from_path(path: Path) -> str:
    stem = path.stem
    if not stem.isdigit():
        raise ValueError(f"FMA ID mode expected a zero-padded numeric filename, got: {path}")
    return str(int(stem))


def _track_id_for_path(path: Path, root: Path, id_mode: str) -> str:
    if id_mode == "relative":
        return local_track_id(path, root)
    if id_mode == "stem":
        return path.stem
    if id_mode == "fma":
        return _fma_id_from_path(path)
    raise ValueError("id_mode must be one of: relative, stem, fma")


def build_local_metadata(
    audio_dir: str | Path,
    id_mode: str = "relative",
) -> tuple[pd.DataFrame, list[Path], list[str]]:
    root = Path(audio_dir).expanduser().resolve()
    files = iter_audio_files(root)
    if not files:
        raise ValueError(f"No supported audio files found below {root}")

    track_ids = [_track_id_for_path(path, root, id_mode=id_mode) for path in files]
    records = [
        TrackRecord(
            track_id=track_id,
            path=str(path),
            title=path.stem,
        ).to_dict()
        for path, track_id in zip(files, track_ids, strict=True)
    ]
    metadata = pd.DataFrame.from_records(records)
    return metadata, files, track_ids


def merge_metadata(base: pd.DataFrame, extra: pd.DataFrame) -> pd.DataFrame:
    if "track_id" not in base.columns or "track_id" not in extra.columns:
        raise ValueError("Both metadata frames must contain a track_id column")
    base = base.copy()
    extra = extra.copy()
    base["track_id"] = base["track_id"].astype(str)
    extra["track_id"] = extra["track_id"].astype(str)
    merged = base.merge(extra, on="track_id", how="left", suffixes=("", "_extra"))
    for col in ["title", "artist", "album", "genre", "split", "subset"]:
        extra_col = f"{col}_extra"
        if extra_col in merged.columns:
            merged[col] = merged[extra_col].combine_first(merged.get(col))
            merged = merged.drop(columns=[extra_col])
    return merged
