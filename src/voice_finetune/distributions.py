"""Module to manage persistent storage of stylometric distributions."""

import json
import pickle
from pathlib import Path
from typing import List

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
DISTRIBUTIONS_DIR = ROOT_DIR / "data" / "distributions"
BASE_COMPLETIONS_FILENAME = "train.jsonl"
FWF_FILENAME = "fwf.pkl"


class DistributionManager:
    """Manages persistent storage and loading of stylometric distributions."""

    def __init__(self, fwf: bool = False) -> None:
        """
        Initialize the distribution manager.

        :param fwf: If True, import function word frequency distribution
        """
        self.fwf = fwf
        self.base_completions: List[dict] = []
        self.fwf_kde = None

        DISTRIBUTIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.files = self._list_distribution_files()
        self._load_base_completions()
        if self.fwf:
            self._load_or_create_fwf()

    @staticmethod
    def _list_distribution_files() -> List[Path]:
        """
        Return a list of distribution files present in the distributions directory.

        :return: List of distribution Path files present in the distributions directory.
        """
        return [f for f in DISTRIBUTIONS_DIR.iterdir() if f.is_file()]

    def _load_base_completions(self) -> None:
        """
        Load base completions from JSONL. If the file does not exist, create it first.

        :return: None
        """
        base_file = DISTRIBUTIONS_DIR / BASE_COMPLETIONS_FILENAME
        if not base_file.exists():
            self._save_base_completions()

        with base_file.open("r", encoding="utf-8") as f:
            self.base_completions = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

    def _load_or_create_fwf(self) -> None:
        """
        Load function word frequency distribution if it exists.

        Otherwise, compute and save it.

        :return: None
        """
        fwf_file = DISTRIBUTIONS_DIR / FWF_FILENAME
        if fwf_file.exists():
            with fwf_file.open("rb") as f:
                self.fwf_kde = pickle.load(f)
        else:
            self._save_fwf_kde()

    @staticmethod
    def _save_base_completions() -> None:
        """
        Save the base completions (train.jsonl) to disk.

        :return: None
        """
        ds = load_dataset("AccelerateScience/bush-dataset", split="train")
        ds.to_json(DISTRIBUTIONS_DIR / BASE_COMPLETIONS_FILENAME)

    def _save_fwf_kde(self) -> None:
        """
        Compute and save the function word frequency distribution of the base completions.

        TODO: Implement this method.
        """
        pass
