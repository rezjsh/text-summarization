# Text Summarization Project (CNN/DailyMail)

An end-to-end, modular **abstractive text summarization** system for news
articles, built on the Kaggle dataset
[`gowrishankarp/newspaper-text-summarization-cnn-dailymail`](https://www.kaggle.com/datasets/gowrishankarp/newspaper-text-summarization-cnn-dailymail)
(CNN/DailyMail article + human-written highlight pairs).

It covers the full lifecycle: Kaggle download → EDA → cleaning/preprocessing
→ configurable seq2seq training (T5 / FLAN-T5 / BART, swappable via a
factory) → ROUGE evaluation → single/batch inference → a Streamlit app.

## Why CNN/DailyMail

- It's the standard benchmark for news summarization: long articles paired
  with short, abstractive "highlight" summaries, so it teaches a model real
  compression and paraphrase, not just extraction.
- It's large (~300k article/summary pairs) but not unwieldy — a few thousand
  examples already produce a coherent summarizer, so it scales cleanly from a
  5-minute test up to a full production-scale fine-tune.
- It ships with train/validation/test splits already separated, so
  evaluation is honest out of the box.

## Project tree

```
text_summarization_project/
├── pyproject.toml            # uv-managed dependencies
├── main.py                   # runs the full pipeline end-to-end
├── Makefile                  # setup / download-data / run-eda / preprocess / train / eval / infer / run-app
├── Dockerfile
├── .env.example              # KAGGLE_USERNAME / KAGGLE_KEY
├── configs/
│   ├── config.yaml            # master config (paths, active model, active subset, generation params...)
│   ├── dataset_config.yaml    # dev / medium / full subset presets
│   ├── model_config.yaml      # model registry: t5-small, flan-t5-small/base, bart-base/large-cnn, pegasus-xsum
│   └── logging_config.yaml
├── data/
│   ├── raw/                  # downloaded + extracted Kaggle csvs
│   ├── interim/
│   └── processed/            # cleaned parquet splits used for training
├── artifacts/
│   ├── eda/                  # eda_report.json, eda_summary.md, PNG plots
│   ├── checkpoints/          # model checkpoints + best_model/
│   └── evaluation/           # evaluation_results.json, predictions_vs_references.csv
├── scripts/                  # thin CLI wrappers around each pipeline stage
│   ├── download_data.py
│   ├── run_eda.py
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── app/
│   ├── streamlit_app.py      # interactive summarizer UI
│   └── explainability.py     # lightweight cross-attention saliency view
├── tests/
│   ├── test_preprocessing.py
│   ├── test_dataset.py
│   └── test_models.py
└── src/text_summarization_project/
    ├── constants/            # static path constants
    ├── entity/                # config_entity.py — typed dataclasses
    ├── config/                # configuration.py — ConfigurationManager
    ├── utils/                 # common.py — yaml/json/logging helpers
    ├── data_ingestion/        # interface.py, strategies.py (Kaggle/local/HF), factory.py, orchestrator.py
    ├── eda/                    # interface.py, strategies.py, visualizers.py, orchestrator.py
    ├── preprocessing/          # interface.py, strategies.py, orchestrator.py
    ├── dataset/                # registry.py (dev/medium/full subsampling), torch_dataset.py
    ├── models/                 # registry.py (model families), factory.py (ModelFactory)
    ├── trainer/                # base_trainer.py (Template Method), seq2seq_trainer.py
    ├── evaluator/              # metrics.py (ROUGE/BERTScore), evaluator.py
    ├── summarizer/             # summarizer.py — Summarizer facade used by inference + app
    ├── inference/               # single.py, batch.py
    └── pipeline/                # stage_01..05_*.py — orchestrate one stage each
```

## Architecture / design patterns

| Pattern             | Where                                                                                                   | Why                                                                                                                               |
| ------------------- | ------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| **Strategy**        | `data_ingestion/strategies.py`, `eda/strategies.py`, `preprocessing/strategies.py`                      | Swap *how* data is fetched/cleaned/analyzed without touching orchestration logic.                                                 |
| **Factory**         | `data_ingestion/factory.py`, `models/factory.py`                                                        | Pick the right ingestion source or model architecture (T5/FLAN-T5/BART/PEGASUS) behind one `.create()` call.                      |
| **Registry**        | `dataset/registry.py`, `models/registry.py`, `configs/dataset_config.yaml`, `configs/model_config.yaml` | Central lookup for dataset subset presets and supported model families — add a new model/subset by editing one yaml + dict entry. |
| **Template Method** | `trainer/base_trainer.py` → `trainer/seq2seq_trainer.py`                                                | Fixed `setup → train → evaluate → save` skeleton; a new training backend only needs to implement the four hooks.                  |
| **Facade**          | `summarizer/summarizer.py`                                                                              | `Summarizer.summarize(text)` hides tokenization/generation/decoding from `inference/` and the Streamlit app.                      |

## Tech stack

Python 3.10+, `uv`, PyTorch, Hugging Face `transformers` + `datasets` +
`evaluate`, `rouge-score`, pandas/numpy, matplotlib/seaborn/plotly, Streamlit,
`kaggle` CLI.

---

## Setup (uv)

```bash
# 1. Install uv if you don't have it: https://docs.astral.sh/uv/getting-started/installation/
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. From the project root:
uv venv
uv pip install -e ".[dev]"

# (equivalent to `make setup`)
```

This creates `.venv/` and installs everything from `pyproject.toml`. Run any
script afterward with `uv run python scripts/<name>.py`.

### Kaggle credentials

```bash
cp .env.example .env
# edit .env and set KAGGLE_USERNAME / KAGGLE_KEY
# (get these from kaggle.com -> Account -> Create New API Token)
```

Alternatively, drop `kaggle.json` at `~/.kaggle/kaggle.json` — either works,
`DataIngestionFactory` checks both.

If neither is available, the pipeline **automatically falls back** to
pulling the equivalent dataset from the Hugging Face Hub
(`abisee/cnn_dailymail`), so the project still runs end-to-end in CI/Colab
without Kaggle credentials.

---

## Running the pipeline

Each stage can be run individually or via `main.py` / `make pipeline`.

```bash
# 1. Download + validate raw data (Kaggle API, falls back to HF Hub)
uv run python scripts/download_data.py --mode auto
# or force a source:
uv run python scripts/download_data.py --mode kaggle
uv run python scripts/download_data.py --mode hf
uv run python scripts/download_data.py --mode local --local_source_dir /path/to/csvs

# 2. EDA — writes artifacts/eda/eda_report.json, eda_summary.md, and PNG plots
uv run python scripts/run_eda.py

# 3. Preprocessing — cleans + filters, writes data/processed/{train,validation,test}.parquet
uv run python scripts/preprocess.py

# 4. Training — fine-tune the configured model on a chosen subset
uv run python scripts/train.py --model flan-t5-small --subset dev

# 5. Evaluation — ROUGE-1/2/L/Lsum + generation stats + latency, on the test split
uv run python scripts/evaluate.py --model_dir artifacts/checkpoints/best_model --subset dev

# 6. Inference
uv run python scripts/infer.py --text "..." --model_dir artifacts/checkpoints/best_model
uv run python scripts/infer.py --input_csv data/raw/cnn_dailymail/test.csv --text_col article \
    --output_csv artifacts/evaluation/batch_summaries.csv --model_dir artifacts/checkpoints/best_model

# 7. Streamlit app
uv run streamlit run app/streamlit_app.py
```

Or all at once:

```bash
uv run python main.py --subset dev            # smoke test end-to-end
uv run python main.py --subset medium --model flan-t5-small
```

`Makefile` shortcuts mirror all of the above: `make setup`, `make
download-data`, `make run-eda`, `make preprocess`, `make train`, `make eval`,
`make infer`, `make run-app`, `make test`, `make pipeline`.

---

## Dataset subsets (dev / medium / full)

Configured in `configs/dataset_config.yaml`, selected via `--subset` or
`dataset.active_subset` in `configs/config.yaml`:

| Subset                 | Train / Val / Test rows        | Use case                                                                                                |
| ---------------------- | ------------------------------ | ------------------------------------------------------------------------------------------------------- |
| `dev`                  | 500 / 100 / 100                | Smoke-test the entire pipeline in minutes, including on CPU.                                            |
| `medium` **(default)** | 20,000 / 2,000 / 2,000         | A real fine-tune that finishes in a single Colab GPU session (~1-3 hrs depending on model size/epochs). |
| `full`                 | ~287k / ~13k / ~11k (all rows) | Production-scale fine-tune; needs a larger GPU and more wall-clock time.                                |

Subsetting happens **after** preprocessing, at load time
(`dataset/registry.py`), so you never need to re-run preprocessing when
switching subsets — only re-run training/evaluation with a different
`--subset`.

## Supported models

Configured in `configs/model_config.yaml`, selected via `--model` or
`model.active_model` in `configs/config.yaml`:

- `t5-small`, `flan-t5-small` (default), `flan-t5-base`
- `bart-base`, `bart-large-cnn`
- `pegasus-xsum`

Add a new model by adding one entry to `configs/model_config.yaml` (and, if
it's a new architecture family, one entry to
`src/text_summarization_project/models/registry.py`).

## EDA outputs

`uv run python scripts/run_eda.py` writes to `artifacts/eda/`:

- `eda_report.json` — dataset overview, missing values, duplicate counts,
  article/summary character & word length stats, approximate token counts,
  top words/n-grams in summaries, per-split row counts, and sample
  article/summary pairs.
- `eda_summary.md` — a short human-readable digest of the above.
- `length_histograms.png`, `article_vs_summary_scatter.png`,
  `top_words_bar.png`, `split_sizes.png`.

## Evaluation outputs

`uv run python scripts/evaluate.py` writes to `artifacts/evaluation/`:

- `evaluation_results.json` — ROUGE-1/2/L/Lsum, generation length stats,
  average per-sample latency, and (if enabled in `config.yaml`) BERTScore.
- `predictions_vs_references.csv` — article (truncated) / reference summary
  / generated summary, for qualitative comparison.

## Tests

```bash
uv run pytest -v
```

Covers preprocessing strategies, dataset subset-registry logic, and the
model-family registry/factory lookup.

## Notes & extension points

- **Mixed precision / gradient accumulation** are already wired through
  `configs/config.yaml` → `training.fp16` / `training.gradient_accumulation_steps`.
- **Early stopping** is enabled via `EarlyStoppingCallback` in
  `trainer/seq2seq_trainer.py`, patience configurable in `config.yaml`.
- **BERTScore** is off by default (slower); flip
  `evaluation.compute_bertscore: true` in `config.yaml` and `uv pip install
  ".[bertscore]"`.
- To add a brand-new pipeline stage (e.g. a distillation step), copy the
  pattern in `pipeline/stage_0X_*.py` and add a `scripts/*.py` wrapper.
