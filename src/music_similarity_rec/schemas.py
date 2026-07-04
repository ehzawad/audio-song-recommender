from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TrackRecord:
    track_id: str
    path: str | None = None
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    genre: str | None = None
    split: str | None = None
    subset: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "path": self.path,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "genre": self.genre,
            "split": self.split,
            "subset": self.subset,
        }


@dataclass(slots=True)
class Recommendation:
    track_id: str
    similarity: float
    distance: float
    rank: int
    path: str | None = None
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    genre: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "track_id": self.track_id,
            "similarity": self.similarity,
            "distance": self.distance,
            "path": self.path,
            "title": self.title,
            "artist": self.artist,
            "album": self.album,
            "genre": self.genre,
        }


def local_track_id(audio_path: Path, audio_root: Path) -> str:
    """Create a readable, stable track ID from a path relative to the corpus root."""
    rel = audio_path.resolve().relative_to(audio_root.resolve()).with_suffix("")
    return rel.as_posix().replace("/", "__")
