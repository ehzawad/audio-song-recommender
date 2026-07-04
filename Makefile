.PHONY: install test toy clean

install:
	python -m pip install --upgrade pip
	python -m pip install -e '.[dev]'

test:
	pytest -q

toy:
	python -m music_similarity_rec.cli make-toy-data --out-dir data/toy_audio
	python -m music_similarity_rec.cli build-audio-index --audio-dir data/toy_audio --artifacts-dir artifacts/toy
	python -m music_similarity_rec.cli recommend-track --artifacts-dir artifacts/toy --track-id a_440 --k 3

clean:
	rm -rf data artifacts .pytest_cache .ruff_cache
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
