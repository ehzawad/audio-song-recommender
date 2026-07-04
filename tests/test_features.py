from __future__ import annotations

import numpy as np

from music_similarity_rec.features import AudioFeatureExtractor, FeatureConfig


def test_extract_array_has_expected_finite_values() -> None:
    sr = 22050
    t = np.linspace(0, 1.0, sr, endpoint=False)
    y = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    extractor = AudioFeatureExtractor(FeatureConfig(duration=None, include_tempo=False))
    features = extractor.extract_array(y, sr)
    assert "mfcc_00_mean" in features
    assert "chroma_00_mean" in features
    assert all(np.isfinite(v) for v in features.values())
