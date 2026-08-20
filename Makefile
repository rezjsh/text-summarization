.PHONY: setup download-data run-eda preprocess train eval infer run-app test pipeline clean

setup:
	uv venv
	uv pip install -e ".[dev]"

download-data:
	uv run python scripts/download_data.py --mode auto

run-eda:
	uv run python scripts/run_eda.py

preprocess:
	uv run python scripts/preprocess.py

train:
	uv run python scripts/train.py --subset dev

train-medium:
	uv run python scripts/train.py --subset medium

eval:
	uv run python scripts/evaluate.py --subset dev --model_dir artifacts/checkpoints/best_model

infer:
	uv run python scripts/infer.py --text "Paste an article here" --model_dir artifacts/checkpoints/best_model

run-app:
	uv run streamlit run app/streamlit_app.py

test:
	uv run pytest -v

pipeline:
	uv run python main.py --subset dev

clean:
	rm -rf artifacts/* logs/* data/interim/* data/processed/*
