from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

from music_similarity_rec.recommender import MusicRecommender


def create_app(artifacts_dir: str | Path) -> FastAPI:
    app = FastAPI(title="Audio Similar-Song Recommender", version="0.1.0")
    recommender = MusicRecommender.load(artifacts_dir)

    @app.get("/health")
    def health() -> dict[str, object]:
        return {"status": "ok", "tracks": len(recommender.index.track_ids)}

    @app.get("/tracks/{track_id}/similar")
    def similar_track(track_id: str, k: int = 10) -> dict[str, object]:
        try:
            df = recommender.recommend_by_track_id(track_id, k=k)
            return {"query": {"track_id": track_id}, "results": recommender.as_records(df)}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/audio/similar")
    async def similar_audio(file: UploadFile = File(...), k: int = 10) -> dict[str, object]:
        suffix = Path(file.filename or "query.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp_path = Path(tmp.name)
            tmp.write(await file.read())
        try:
            df = recommender.recommend_by_audio_file(tmp_path, k=k)
            return {"query": {"filename": file.filename}, "results": recommender.as_records(df)}
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        finally:
            tmp_path.unlink(missing_ok=True)

    return app
