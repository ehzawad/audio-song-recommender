from __future__ import annotations

from pathlib import Path

from music_similarity_rec.features import AudioFeatureExtractor, FeatureConfig
from music_similarity_rec.index import SimilarityIndex
from music_similarity_rec.metadata import build_local_metadata
from music_similarity_rec.recommender import MusicRecommender
from music_similarity_rec.toydata import create_toy_audio


def test_toy_pipeline_end_to_end(tmp_path: Path) -> None:
    audio_dir = create_toy_audio(tmp_path / "audio", seconds=1.0)
    metadata, files, track_ids = build_local_metadata(audio_dir)
    extractor = AudioFeatureExtractor(FeatureConfig(duration=None, include_tempo=False))
    features, errors = extractor.extract_many(files, track_ids, show_progress=False)
    assert errors.empty

    index = SimilarityIndex.build(features)
    artifacts = tmp_path / "artifacts"
    index.save(artifacts, features, metadata)

    recommender = MusicRecommender.load(artifacts, extractor_config=FeatureConfig(duration=None, include_tempo=False))
    results = recommender.recommend_by_track_id("a_440", k=2)
    assert len(results) == 2
    assert "track_id" in results.columns
