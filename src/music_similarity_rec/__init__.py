"""Audio-based similar-song recommendation toolkit."""

from music_similarity_rec.features import AudioFeatureExtractor, FeatureConfig
from music_similarity_rec.index import SimilarityIndex
from music_similarity_rec.recommender import MusicRecommender

__all__ = [
    "AudioFeatureExtractor",
    "FeatureConfig",
    "SimilarityIndex",
    "MusicRecommender",
]

__version__ = "0.1.0"
