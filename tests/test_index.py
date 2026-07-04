from __future__ import annotations

import pandas as pd

from music_similarity_rec.index import SimilarityIndex


def test_similarity_index_returns_nearest_neighbor_without_self() -> None:
    features = pd.DataFrame(
        {
            "track_id": ["a", "b", "c"],
            "x": [0.0, 0.1, 10.0],
            "y": [1.0, 1.1, -5.0],
        }
    )
    index = SimilarityIndex.build(features)
    results = index.query_track(features, "a", k=1)
    assert results.iloc[0]["track_id"] == "b"
