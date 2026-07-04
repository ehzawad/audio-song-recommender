from __future__ import annotations

from pathlib import Path

SUPPORTED_AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}


def iter_audio_files(root: str | Path) -> list[Path]:
    root_path = Path(root).expanduser().resolve()
    if not root_path.exists():
        raise FileNotFoundError(f"Audio directory does not exist: {root_path}")
    if not root_path.is_dir():
        raise NotADirectoryError(f"Expected an audio directory, got: {root_path}")

    files = [
        path
        for path in root_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS
    ]
    return sorted(files)


def ensure_dir(path: str | Path) -> Path:
    out = Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out
