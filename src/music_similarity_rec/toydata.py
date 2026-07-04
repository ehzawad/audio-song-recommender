from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from music_similarity_rec.paths import ensure_dir


def _tone(freq: float, sr: int, seconds: float, phase: float = 0.0) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    envelope = np.linspace(0.15, 1.0, t.size)
    y = 0.4 * np.sin(2 * np.pi * freq * t + phase)
    y += 0.08 * np.sin(2 * np.pi * (freq * 2.0) * t + phase / 2)
    return (y * envelope).astype(np.float32)


def create_toy_audio(out_dir: str | Path, sr: int = 22_050, seconds: float = 4.0) -> Path:
    """Create a tiny synthetic corpus for smoke-testing the full pipeline."""
    out = ensure_dir(out_dir)
    specs = {
        "a_440.wav": (440.0, 0.0),
        "a_445.wav": (445.0, 0.1),
        "b_880.wav": (880.0, 0.0),
        "b_890.wav": (890.0, 0.2),
        "c_1320.wav": (1320.0, 0.0),
        "c_1335.wav": (1335.0, 0.3),
    }
    for name, (freq, phase) in specs.items():
        sf.write(out / name, _tone(freq, sr, seconds, phase=phase), sr)
    return out
