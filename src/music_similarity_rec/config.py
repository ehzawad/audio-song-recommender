from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class FeatureExtractionSettings:
    sample_rate: int = 22_050
    duration: float | None = 30.0
    offset: float = 0.0
    n_fft: int = 2048
    hop_length: int = 512
    n_mfcc: int = 20
    include_tempo: bool = True


@dataclass(slots=True)
class IndexSettings:
    metric: str = "cosine"
    algorithm: str = "brute"
    pca_components: int | None = None
    random_state: int = 42


@dataclass(slots=True)
class RecommendationSettings:
    default_k: int = 10


@dataclass(slots=True)
class AppConfig:
    feature_extraction: FeatureExtractionSettings = field(default_factory=FeatureExtractionSettings)
    index: IndexSettings = field(default_factory=IndexSettings)
    recommendation: RecommendationSettings = field(default_factory=RecommendationSettings)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge_dict(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load a YAML config file, falling back to dataclass defaults."""
    defaults = AppConfig().to_dict()
    if path is None:
        data = defaults
    else:
        with Path(path).open("r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        data = _merge_dict(defaults, loaded)

    return AppConfig(
        feature_extraction=FeatureExtractionSettings(**data.get("feature_extraction", {})),
        index=IndexSettings(**data.get("index", {})),
        recommendation=RecommendationSettings(**data.get("recommendation", {})),
    )
