from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from music_similarity_rec.config import FeatureExtractionSettings


@dataclass(slots=True)
class FeatureConfig:
    sample_rate: int = 22_050
    duration: float | None = 30.0
    offset: float = 0.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mfcc: int = 20
    include_tempo: bool = True

    @classmethod
    def from_settings(cls, settings: FeatureExtractionSettings) -> "FeatureConfig":
        return cls(**asdict(settings))


class AudioFeatureExtractor:
    """Extract compact, hand-engineered MIR features from a music file.

    The resulting vector is intentionally classical and transparent: MFCCs,
    chroma, spectral shape, RMS energy, zero-crossing rate, and optional tempo.
    It is a solid baseline for content-based recommendation and an easy place to
    swap in neural embeddings later.
    """

    def __init__(self, config: FeatureConfig | None = None) -> None:
        self.config = config or FeatureConfig()

    def load_audio(self, path: str | Path) -> tuple[np.ndarray, int]:
        cfg = self.config
        y, sr = librosa.load(
            path,
            sr=cfg.sample_rate,
            mono=True,
            offset=cfg.offset,
            duration=cfg.duration,
        )
        if y.size == 0:
            raise ValueError(f"No audio samples could be loaded from {path}")
        return y.astype(np.float32, copy=False), sr

    @staticmethod
    def _safe_stats(values: np.ndarray, prefix: str) -> dict[str, float]:
        arr = np.asarray(values, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)

        # Replace non-finite values before summarization. This is pragmatic for
        # damaged files and silent sections; the imputer in the index is a second
        # line of defense.
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        eps = 1e-12

        mean = arr.mean(axis=1)
        std = arr.std(axis=1)
        centered = arr - mean[:, None]
        standardized = centered / (std[:, None] + eps)
        skew = np.mean(standardized**3, axis=1)
        kurtosis = np.mean(standardized**4, axis=1) - 3.0

        stats = {
            "mean": mean,
            "std": std,
            "median": np.median(arr, axis=1),
            "min": arr.min(axis=1),
            "max": arr.max(axis=1),
            "skew": skew,
            "kurtosis": kurtosis,
        }

        out: dict[str, float] = {}
        for stat_name, stat_values in stats.items():
            for i, value in enumerate(stat_values):
                out[f"{prefix}_{i:02d}_{stat_name}"] = float(value)
        return out

    def extract_array(self, y: np.ndarray, sr: int) -> dict[str, float]:
        cfg = self.config
        features: dict[str, float] = {}

        mfcc = librosa.feature.mfcc(
            y=y,
            sr=sr,
            n_mfcc=cfg.n_mfcc,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        features.update(self._safe_stats(mfcc, "mfcc"))

        chroma = librosa.feature.chroma_stft(
            y=y,
            sr=sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        features.update(self._safe_stats(chroma, "chroma"))

        spectral_centroid = librosa.feature.spectral_centroid(
            y=y,
            sr=sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        features.update(self._safe_stats(spectral_centroid, "spectral_centroid"))

        spectral_bandwidth = librosa.feature.spectral_bandwidth(
            y=y,
            sr=sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        features.update(self._safe_stats(spectral_bandwidth, "spectral_bandwidth"))

        spectral_contrast = librosa.feature.spectral_contrast(
            y=y,
            sr=sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        features.update(self._safe_stats(spectral_contrast, "spectral_contrast"))

        spectral_rolloff = librosa.feature.spectral_rolloff(
            y=y,
            sr=sr,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        features.update(self._safe_stats(spectral_rolloff, "spectral_rolloff"))

        rms = librosa.feature.rms(
            y=y,
            frame_length=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        features.update(self._safe_stats(rms, "rms"))

        zcr = librosa.feature.zero_crossing_rate(
            y=y,
            frame_length=cfg.n_fft,
            hop_length=cfg.hop_length,
        )
        features.update(self._safe_stats(zcr, "zero_crossing_rate"))

        if cfg.include_tempo:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=cfg.hop_length)
            tempo_arr = librosa.feature.tempo(
                onset_envelope=onset_env,
                sr=sr,
                hop_length=cfg.hop_length,
            )
            features["tempo_00_mean"] = float(np.ravel(tempo_arr)[0])

        return features

    def extract_file(self, path: str | Path) -> dict[str, float]:
        y, sr = self.load_audio(path)
        return self.extract_array(y, sr)

    def extract_many(
        self,
        files: Iterable[Path],
        track_ids: Iterable[str],
        show_progress: bool = True,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        file_list = list(files)
        id_list = list(track_ids)
        if len(file_list) != len(id_list):
            raise ValueError("files and track_ids must have the same length")
        if not file_list:
            raise ValueError("No audio files were provided for feature extraction")

        records: list[dict[str, float | str]] = []
        errors: list[dict[str, str]] = []
        iterator = zip(file_list, id_list, strict=True)
        if show_progress:
            iterator = tqdm(list(iterator), desc="extracting audio features")

        for path, track_id in iterator:
            try:
                feature_row = self.extract_file(path)
                feature_row["track_id"] = track_id
                records.append(feature_row)
            except Exception as exc:  # noqa: BLE001 - keep batch extraction resilient.
                errors.append({"track_id": track_id, "path": str(path), "error": repr(exc)})

        if not records:
            raise RuntimeError(f"Feature extraction failed for all {len(file_list)} files")

        features = pd.DataFrame.from_records(records).set_index("track_id").sort_index()
        error_df = pd.DataFrame.from_records(errors, columns=["track_id", "path", "error"])
        return features, error_df
