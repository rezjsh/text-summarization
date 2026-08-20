"""Concrete data ingestion strategies.

KaggleAPIStrategy   -> downloads via Kaggle API using requests (requires
                       KAGGLE_API_TOKEN env var, see .env.example)
LocalCopyStrategy    -> for when the user already has the csvs on disk
                       (e.g. manually downloaded, or mounted in Colab)
HFDatasetsStrategy   -> fallback that pulls the equivalent dataset from the
                       Hugging Face Hub (`abisee/cnn_dailymail`) when Kaggle
                       credentials aren't available -- handy for CI / Colab.
"""
import logging
import os
import shutil
import zipfile
from pathlib import Path
from typing import List

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

from text_summarization_project.data_ingestion.interface import DataIngestionStrategy

logger = logging.getLogger(__name__)


class KaggleAPIStrategy(DataIngestionStrategy):
    def __init__(self, kaggle_dataset: str, raw_dir: Path, unzip_dir: Path):
        self.kaggle_dataset = kaggle_dataset
        self.raw_dir = Path(raw_dir)
        self.unzip_dir = Path(unzip_dir)

    def _build_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(
            total=5,
            connect=5,
            read=5,
            backoff_factor=2,
            status_forcelist=(429, 500, 502, 503, 504),
            allowed_methods=frozenset(["GET"]),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry)
        session.mount("https://", adapter)
        session.mount("http://", adapter)
        return session

    def fetch(self) -> Path:
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.unzip_dir.mkdir(parents=True, exist_ok=True)

        zip_name = self.kaggle_dataset.split("/")[-1] + ".zip"
        zip_path = self.raw_dir / zip_name

        if not zip_path.exists() or zip_path.stat().st_size == 0:
            logger.info("Starting Kaggle token-based download...")

            token = os.getenv("KAGGLE_API_TOKEN")
            if not token:
                raise EnvironmentError("KAGGLE_API_TOKEN is missing")

            dataset_slug = self.kaggle_dataset.strip()
            owner, name = dataset_slug.split("/", 1)
            url = f"https://www.kaggle.com/api/v1/datasets/download/{owner}/{name}"
            headers = {"Authorization": f"Bearer {token}"}

            tmp_file = zip_path.with_suffix(".part")
            session = self._build_session()

            try:
                with session.get(url, headers=headers, stream=True, timeout=(30, 300), allow_redirects=True) as response:
                    response.raise_for_status()
                    total = int(response.headers.get("content-length", 0))
                    with open(tmp_file, "wb") as f, tqdm(
                       total=total,
                       unit="B",
                       unit_scale=True,
                       desc="Downloading Kaggle dataset"
                    ) as pbar:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            if chunk:
                                f.write(chunk)
                                pbar.update(len(chunk))

                tmp_file.replace(zip_path)
                logger.info(f"Dataset downloaded successfully to {zip_path}")

            except requests.exceptions.ChunkedEncodingError as e:
                if tmp_file.exists():
                    tmp_file.unlink(missing_ok=True)
                raise RuntimeError(
                    "Kaggle download was interrupted mid-transfer. "
                    "Try again, use a stable network, or switch to manual/local ingestion."
                ) from e
            except Exception:
                if tmp_file.exists():
                    tmp_file.unlink(missing_ok=True)
                raise
        else:
            logger.info(f"Zip already present at {zip_path}, skipping download.")

        logger.info(f"Extracting {zip_path} -> {self.unzip_dir}")
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(self.unzip_dir)
        except zipfile.BadZipFile as e:
            raise RuntimeError(f"Downloaded file is not a valid zip archive: {e}") from e

        return self.unzip_dir

    def validate(self, data_dir: Path, expected_files: List[str]) -> bool:
        data_dir = Path(data_dir)
        ok = True
        for fname in expected_files:
            fpath = KaggleAPIStrategy._find_file(data_dir, fname)
            if fpath is None or fpath.stat().st_size == 0:
                logger.error(f"Missing or empty expected file: {fname}")
                ok = False
            else:
                logger.info(f"Validated {fname} -> {fpath} ({fpath.stat().st_size} bytes)")
        return ok

    @staticmethod
    def _find_file(data_dir: Path, fname: str):
        matches = list(Path(data_dir).rglob(fname))
        return matches[0] if matches else None


class LocalCopyStrategy(DataIngestionStrategy):
    """Copies csvs from an arbitrary source directory (e.g. a manual Kaggle
    download unzipped elsewhere, or a Colab-mounted Drive folder)."""

    def __init__(self, source_dir: Path, unzip_dir: Path):
        self.source_dir = Path(source_dir)
        self.unzip_dir = Path(unzip_dir)

    def fetch(self) -> Path:
        self.unzip_dir.mkdir(parents=True, exist_ok=True)
        for csv_file in self.source_dir.glob("*.csv"):
            dest = self.unzip_dir / csv_file.name
            if not dest.exists():
                shutil.copy2(csv_file, dest)
                logger.info(f"Copied {csv_file} -> {dest}")
        return self.unzip_dir

    def validate(self, data_dir: Path, expected_files: List[str]) -> bool:
        return KaggleAPIStrategy.validate(self, data_dir, expected_files)


class HFDatasetsStrategy(DataIngestionStrategy):
    """Fallback: pulls the equivalent dataset from Hugging Face Hub and
    writes it out as train.csv / validation.csv / test.csv so the rest of
    the pipeline never needs to know which source was used."""

    def __init__(self, unzip_dir: Path, hf_dataset: str = "abisee/cnn_dailymail", hf_config: str = "3.0.0"):
        self.unzip_dir = Path(unzip_dir)
        self.hf_dataset = hf_dataset
        self.hf_config = hf_config

    def fetch(self) -> Path:
        from datasets import load_dataset

        self.unzip_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Loading '{self.hf_dataset}' ({self.hf_config}) from the Hugging Face Hub ...")
        ds = load_dataset(self.hf_dataset, self.hf_config)

        split_to_file = {"train": "train.csv", "validation": "validation.csv", "test": "test.csv"}
        for split, fname in split_to_file.items():
            out_path = self.unzip_dir / fname
            if not out_path.exists():
                df = ds[split].to_pandas()
                df = df.rename(columns={"article": "article", "highlights": "highlights"})
                df.to_csv(out_path, index=False)
                logger.info(f"Wrote {out_path} ({len(df)} rows)")
        return self.unzip_dir

    def validate(self, data_dir: Path, expected_files: List[str]) -> bool:
        return KaggleAPIStrategy.validate(self, data_dir, expected_files)