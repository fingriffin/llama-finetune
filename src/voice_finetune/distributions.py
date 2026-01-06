"""Module to manage persistent storage of stylometric distributions."""

import json
import pickle
import string
from pathlib import Path
from typing import List

import nltk
import numpy as np
from datasets import load_dataset
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from scipy.stats import gaussian_kde

ROOT_DIR = Path(__file__).resolve().parents[2]
DISTRIBUTIONS_DIR = ROOT_DIR / "data" / "distributions"
BASE_COMPLETIONS_FILENAME = "train.jsonl"
FWF_FILENAME = "fwf.pkl"


class DistributionManager:
    """Manages persistent storage and loading of stylometric distributions."""

    def __init__(self,
                 *,
                 fwf: bool = False
        ) -> None:
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
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
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
        Load function word frequency KDE if it exists.

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

        :return: None
        """
        fwfs = [
            self.calculate_fwf(c["messages"][2]["content"]) for c in self.base_completions
        ]

        if not fwfs:
            raise ValueError("No function word frequencies computed. "
                             "Base completions may be empty.")

        # Fit KDE to the scalar FWF values
        kde = gaussian_kde(np.array(fwfs))

        # Store in instance
        self.fwf_kde = kde

        # Save KDE to disk
        fwf_file = DISTRIBUTIONS_DIR / FWF_FILENAME
        with fwf_file.open("wb") as f:
            pickle.dump(kde, f)

    @staticmethod
    def calculate_fwf(completion: str) -> float:
        """
        Compute the function word frequency of a completion.

        :param completion: a plain text completion of a prompt.
        :return: function word frequency of the completion.
        """
        stop_words = set(stopwords.words("english"))

        words = word_tokenize(completion.lower())

        words_no_punct = [word for word in words if word not in string.punctuation]

        fw_count = sum(1 for word in words_no_punct if word in stop_words)

        return fw_count / len(words_no_punct) if words_no_punct else 0
