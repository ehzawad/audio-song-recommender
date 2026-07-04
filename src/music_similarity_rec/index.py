from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from music_similarity_rec.config import IndexSettings
from music_similarity_rec.paths import ensure_dir


@dataclass(slots=True)
class IndexConfig:
    metric: str = "cosine"
    algorithm: str = "brute"
    pca_components: int | None = None
    random_state: int = 42

    @classmethod
    def from_settings(cls, settings: IndexSettings) -> "IndexConfig":
        return cls(**asdict(settings))


class SimilarityIndex:
    """A serializable k-nearest-neighbor index over song feature vectors."""

    ARTIFACT_NAME = "index.joblib"
    FEATURES_NAME = "features.csv"
    METADATA_NAME = "metadata.csv"

    def __init__(
        self,
        preprocessor: Pipeline,
        nearest_neighbors: NearestNeighbors,
        feature_columns: list[str],
        track_ids: list[str],
        config: IndexConfig,
    ) -> None:
        self.preprocessor = preprocessor
        self.nearest_neighbors = nearest_neighbors
        self.feature_columns = feature_columns
        self.track_ids = track_ids
        self.config = config
        self._id_to_pos = {track_id: i for i, track_id in enumerate(track_ids)}

    @staticmethod
    def _clean_features(features: pd.DataFrame) -> pd.DataFrame:
        if "track_id" in features.columns:
            features = features.set_index("track_id")
        features = features.copy()
        features.index = features.index.astype(str)
        numeric = features.apply(pd.to_numeric, errors="coerce")
        numeric = numeric.replace([np.inf, -np.inf], np.nan)
        # Drop columns that are entirely missing because SimpleImputer cannot
        # learn useful statistics for them.
        numeric = numeric.dropna(axis=1, how="all")
        if numeric.empty:
            raise ValueError("No numeric feature columns remain after cleaning")
        return numeric.sort_index()

    @classmethod
    def build(
        cls,
        features: pd.DataFrame,
        config: IndexConfig | None = None,
    ) -> "SimilarityIndex":
        cfg = config or IndexConfig()
        clean = cls._clean_features(features)
        track_ids = clean.index.astype(str).tolist()
        n_samples, n_features = clean.shape
        if n_samples < 2:
            raise ValueError("At least two tracks are required to build a recommendation index")

        steps: list[tuple[str, Any]] = [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
        if cfg.pca_components is not None:
            max_components = min(n_samples, n_features)
            if not 1 <= cfg.pca_components <= max_components:
                raise ValueError(
                    "pca_components must be between 1 and "
                    f"{max_components}; got {cfg.pca_components}"
                )
            steps.append(
                (
                    "pca",
                    PCA(n_components=cfg.pca_components, random_state=cfg.random_state),
                )
            )

        preprocessor = Pipeline(steps)
        transformed = preprocessor.fit_transform(clean)
        nn = NearestNeighbors(metric=cfg.metric, algorithm=cfg.algorithm)
        nn.fit(transformed)

        return cls(
            preprocessor=preprocessor,
            nearest_neighbors=nn,
            feature_columns=clean.columns.astype(str).tolist(),
            track_ids=track_ids,
            config=cfg,
        )

    def _align_query(self, query: pd.DataFrame | pd.Series | dict[str, float]) -> pd.DataFrame:
        if isinstance(query, dict):
            query_df = pd.DataFrame([query])
        elif isinstance(query, pd.Series):
            query_df = query.to_frame().T
        else:
            query_df = query.copy()

        query_df = query_df.apply(pd.to_numeric, errors="coerce")
        aligned = query_df.reindex(columns=self.feature_columns)
        missing = aligned.columns[aligned.isna().all(axis=0)].tolist()
        # Missing columns are allowed; the fitted imputer fills them. But if a
        # user sends a totally incompatible vector, fail loudly.
        if len(missing) == len(self.feature_columns):
            raise ValueError(
                "The query vector has none of the feature columns expected by the index. "
                "Build the index and query vector with the same feature extractor."
            )
        return aligned

    def query_vector(self, query: pd.DataFrame | pd.Series | dict[str, float], k: int = 10) -> pd.DataFrame:
        if k < 1:
            raise ValueError("k must be >= 1")
        aligned = self._align_query(query)
        transformed = self.preprocessor.transform(aligned)
        n_neighbors = min(k, len(self.track_ids))
        distances, indices = self.nearest_neighbors.kneighbors(
            transformed,
            n_neighbors=n_neighbors,
            return_distance=True,
        )
        rows: list[dict[str, Any]] = []
        for rank, (distance, idx) in enumerate(zip(distances[0], indices[0], strict=True), start=1):
            rows.append(
                {
                    "rank": rank,
                    "track_id": self.track_ids[int(idx)],
                    "distance": float(distance),
                    "similarity": float(1.0 - distance) if self.config.metric == "cosine" else float(-distance),
                }
            )
        return pd.DataFrame(rows)

    def query_track(self, features: pd.DataFrame, track_id: str, k: int = 10) -> pd.DataFrame:
        track_id = str(track_id)
        clean = self._clean_features(features)
        if track_id not in clean.index:
            raise KeyError(f"Unknown track_id: {track_id}")
        # Ask for one extra neighbor so we can remove the query track itself.
        raw = self.query_vector(clean.loc[track_id], k=min(k + 1, len(self.track_ids)))
        raw = raw[raw["track_id"] != track_id].head(k).copy()
        raw["rank"] = range(1, len(raw) + 1)
        return raw

    def save(self, artifacts_dir: str | Path, features: pd.DataFrame, metadata: pd.DataFrame) -> Path:
        out = ensure_dir(artifacts_dir)
        joblib.dump(
            {
                "preprocessor": self.preprocessor,
                "nearest_neighbors": self.nearest_neighbors,
                "feature_columns": self.feature_columns,
                "track_ids": self.track_ids,
                "config": asdict(self.config),
            },
            out / self.ARTIFACT_NAME,
        )
        self._clean_features(features).to_csv(out / self.FEATURES_NAME, index_label="track_id")
        metadata.copy().to_csv(out / self.METADATA_NAME, index=False)
        return out

    @classmethod
    def load(cls, artifacts_dir: str | Path) -> "SimilarityIndex":
        root = Path(artifacts_dir).expanduser().resolve()
        payload = joblib.load(root / cls.ARTIFACT_NAME)
        return cls(
            preprocessor=payload["preprocessor"],
            nearest_neighbors=payload["nearest_neighbors"],
            feature_columns=list(payload["feature_columns"]),
            track_ids=[str(x) for x in payload["track_ids"]],
            config=IndexConfig(**payload["config"]),
        )
