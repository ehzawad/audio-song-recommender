from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from music_similarity_rec.features import AudioFeatureExtractor, FeatureConfig
from music_similarity_rec.index import SimilarityIndex


class MusicRecommender:
    """Load saved artifacts and return metadata-enriched recommendations."""

    def __init__(
        self,
        index: SimilarityIndex,
        features: pd.DataFrame,
        metadata: pd.DataFrame,
        extractor: AudioFeatureExtractor | None = None,
    ) -> None:
        self.index = index
        self.features = SimilarityIndex._clean_features(features)
        self.metadata = metadata.copy()
        self.metadata["track_id"] = self.metadata["track_id"].astype(str)
        self.extractor = extractor or AudioFeatureExtractor()

    @classmethod
    def load(
        cls,
        artifacts_dir: str | Path,
        extractor_config: FeatureConfig | None = None,
    ) -> "MusicRecommender":
        root = Path(artifacts_dir).expanduser().resolve()
        index = SimilarityIndex.load(root)
        features = pd.read_csv(root / SimilarityIndex.FEATURES_NAME, dtype={"track_id": str})
        metadata = pd.read_csv(root / SimilarityIndex.METADATA_NAME, dtype={"track_id": str})
        extractor = AudioFeatureExtractor(extractor_config or FeatureConfig())
        return cls(index=index, features=features, metadata=metadata, extractor=extractor)

    def _join_metadata(self, results: pd.DataFrame) -> pd.DataFrame:
        merged = results.merge(self.metadata, on="track_id", how="left")
        wanted = [
            "rank",
            "track_id",
            "similarity",
            "distance",
            "title",
            "artist",
            "album",
            "genre",
            "path",
            "split",
            "subset",
        ]
        return merged[[col for col in wanted if col in merged.columns]]

    def recommend_by_track_id(self, track_id: str, k: int = 10) -> pd.DataFrame:
        results = self.index.query_track(self.features, str(track_id), k=k)
        return self._join_metadata(results)

    def recommend_by_audio_file(self, audio_path: str | Path, k: int = 10) -> pd.DataFrame:
        vector = self.extractor.extract_file(audio_path)
        results = self.index.query_vector(vector, k=k)
        return self._join_metadata(results)

    @staticmethod
    def as_records(df: pd.DataFrame) -> list[dict[str, Any]]:
        clean = df.where(pd.notna(df), None)
        return clean.to_dict(orient="records")
