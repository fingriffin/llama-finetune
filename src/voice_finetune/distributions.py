"""Module to manage persistent storage of stylometric distributions."""

import json
import pickle
import re
import string
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
DISTRIBUTIONS_DIR = ROOT_DIR / "data" / "distributions"
TRAIN_FILENAME = "train.jsonl"

FWF_FILENAME_TRUE = "fwf_true.pkl"
FWF_FILENAME_BASE = "fwf_base.pkl"

MATTR_FILENAME_TRUE = "mattr_true.pkl"
MATTR_FILENAME_BASE = "mattr_base.pkl"

HAPAX_FILENAME_TRUE = "hapax_true.pkl"
HAPAX_FILENAME_BASE = "hapax_base.pkl"

# TODO: Move this
STOP_WORDS = [
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
    "you", "your", "yours", "yourself", "yourselves", "he", "him",
    "his", "himself", "she", "her", "hers", "herself", "it", "its",
    "itself", "they", "them", "their", "theirs", "themselves",
    "what", "which", "who", "whom", "this", "that", "these", "those",
    "am", "is", "are", "was", "were", "be", "been", "being", "have",
    "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until",
    "while", "of", "at", "by", "for", "with", "about", "against",
    "between", "into", "through", "during", "before", "after",
    "above", "below", "to", "from", "up", "down", "in", "out", "on",
    "off", "over", "under", "again", "further", "then", "once", "here",
    "there", "when", "where", "why", "how", "all", "any", "both", "each",
    "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now"
]

class DistributionManager:
    """Manages persistent storage and loading of stylometric distributions."""

    def __init__(self,
                 *,
                 fwf: bool = False,
                 mattr: bool = False,
                 hapax: bool = False,
        ) -> None:
        """
        Initialize the distribution manager.

        :param fwf: If True, import function word frequency distribution
        :param mattr: If True, import MATTR distribution
        :param hapax: If True, import hapax legomena distribution
        """
        self.fwf = fwf
        self.mattr = mattr
        self.hapax = hapax

        self.true_completions: List[dict] = []
        self.base_completions: List[dict] = []

        self.fwf_true_sorted: np.ndarray | None = None
        self.fwf_base_sorted: np.ndarray | None = None

        self.mattr_true_sorted: np.ndarray | None = None
        self.mattr_base_sorted: np.ndarray | None = None

        self.hapax_true_sorted: np.ndarray | None = None
        self.hapax_base_sorted: np.ndarray | None = None

        DISTRIBUTIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.files = self._list_distribution_files()
        self._load_completions()

        if self.fwf:
            self._load_or_create_fwf()

        if self.mattr:
            self._load_or_create_mattr()

        if self.hapax:
            self._load_or_create_hapax()

    @staticmethod
    def _list_distribution_files() -> List[Path]:
        """
        Return a list of distribution files present in the distributions directory.

        :return: List of distribution Path files present in the distributions directory.
        """
        return [f for f in DISTRIBUTIONS_DIR.iterdir() if f.is_file()]

    def _load_completions(self) -> None:
        """
        Load static completions from JSONL. If the files do not exist, create them first.

        Static completions refer to the true completions and base model generations.

        :return: None
        """
        true_file = DISTRIBUTIONS_DIR / "true" / TRAIN_FILENAME
        base_file = DISTRIBUTIONS_DIR / "base" / TRAIN_FILENAME
        if not true_file.exists():
            self._save_true_completions()
        if not base_file.exists():
            self._save_base_completions()

        with true_file.open("r", encoding="utf-8") as f:
            self.true_completions = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        with base_file.open("r", encoding="utf-8") as f:
            self.base_completions = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

    def _load_or_create_fwf(self) -> None:
        """
        Load function word frequency empirical CDFs if they exist.

        Otherwise, compute and save them.

        :return: None
        """
        fwf_file_true = DISTRIBUTIONS_DIR / "true" / FWF_FILENAME_TRUE
        fwf_file_base = DISTRIBUTIONS_DIR / "base" / FWF_FILENAME_BASE
        if fwf_file_true.exists() and fwf_file_base.exists():
            with fwf_file_true.open("rb") as f:
                self.fwf_true_sorted = pickle.load(f)
            with fwf_file_base.open("rb") as f:
                self.fwf_base_sorted = pickle.load(f)
        else:
            self._save_fwf_cdf()

    def _load_or_create_mattr(self) -> None:
        """
        Load MATTR (Moving average TTR) empirical CDFs if they exist.

        Otherwise, compute and save them.

        :return: None
        """
        mattr_file_true = DISTRIBUTIONS_DIR / "true" / MATTR_FILENAME_TRUE
        mattr_file_base = DISTRIBUTIONS_DIR / "base" / MATTR_FILENAME_BASE

        if mattr_file_true.exists() and mattr_file_base.exists():
            with mattr_file_true.open("rb") as f:
                self.mattr_true_sorted = pickle.load(f)

            with mattr_file_base.open("rb") as f:
                self.mattr_base_sorted = pickle.load(f)
        else:
            self._save_mattr_cdf()

    def _load_or_create_hapax(self) -> None:
        """
        Load hapax legomena empirical CDFs if they exist.

        Otherwise, compute and save them.

        :return: None
        """
        hapax_file_true = DISTRIBUTIONS_DIR / "true" / HAPAX_FILENAME_TRUE
        hapax_file_base = DISTRIBUTIONS_DIR / "base" / HAPAX_FILENAME_BASE

        if hapax_file_true.exists() and hapax_file_base.exists():
            with hapax_file_true.open("rb") as f:
                self.hapax_true_sorted = pickle.load(f)

            with hapax_file_base.open("rb") as f:
                self.hapax_base_sorted = pickle.load(f)
        else:
            self._save_hapax_cdf()

    @staticmethod
    def _save_true_completions() -> None:
        """
        Save the true completions (train.jsonl) to disk.

        :return: None
        """
        ds = load_dataset(
            "AccelerateScience/bush-dataset", # TODO: Config controlled
            split="train"
        )
        ds.to_json(DISTRIBUTIONS_DIR / "true" / TRAIN_FILENAME)

    @staticmethod
    def _save_base_completions() -> None:
        """
        Save the base completions (base_*.jsonl) to disk.

        * is the number of parameters

        :return: None
        """
        ds = load_dataset(
            "AccelerateScience/bush-base-completions-8b", # TODO: Config controlled
            split="train"
        )
        ds.to_json(DISTRIBUTIONS_DIR / "base" / TRAIN_FILENAME)

    def _save_fwf_cdf(self) -> None:
        """
        Compute and save the fwf distributions of the static completions.

        :return: None
        """
        fwfs_true = [
            self.calculate_fwf(c["messages"][2]["content"]) for c in self.true_completions
        ]

        fwfs_base = [
            self.calculate_fwf(c["messages"][2]["content"]) for c in self.base_completions
        ]

        if not fwfs_true:
            raise ValueError("No function word frequencies computed. "
                             "True completions may be empty.")

        if not fwfs_base:
            raise ValueError("No function word frequencies computed. "
                             "BASE completions may be empty.")

        self.fwf_true_sorted = np.sort(np.asarray(fwfs_true, dtype=np.float64))
        self.fwf_base_sorted = np.sort(np.asarray(fwfs_base, dtype=np.float64))

        fwf_file_true = DISTRIBUTIONS_DIR / "true" / FWF_FILENAME_TRUE
        with fwf_file_true.open("wb") as f:
            pickle.dump(self.fwf_true_sorted, f)

        fwf_file_base = DISTRIBUTIONS_DIR / "base" / FWF_FILENAME_BASE
        with fwf_file_base.open("wb") as f:
            pickle.dump(self.fwf_base_sorted, f)

    def _save_mattr_cdf(self) -> None:
        """
        Compute and save the MATTR distributions of the static completions.

        :return: None
        """
        mattrs_true = [
            self.calculate_mattr(
                c["messages"][2]["content"])
            for c in self.true_completions
        ]

        mattrs_base = [
            self.calculate_mattr(
                c["messages"][2]["content"])
            for c in self.base_completions
        ]

        if not mattrs_true:
            raise ValueError("No MATTR values computed. "
                             "True completions may be empty.")

        if not mattrs_base:
            raise ValueError("No MATTR values computed. "
                             "Base completions may be empty.")

        self.mattr_true_sorted = np.sort(np.asarray(mattrs_true, dtype=np.float64))
        self.mattr_base_sorted = np.sort(np.asarray(mattrs_base, dtype=np.float64))

        mattr_file_true = DISTRIBUTIONS_DIR / "true" / MATTR_FILENAME_TRUE
        mattr_file_base = DISTRIBUTIONS_DIR / "base" / MATTR_FILENAME_BASE
        with mattr_file_true.open("wb") as f:
            pickle.dump(self.mattr_true_sorted, f)
        with mattr_file_base.open("wb") as f:
            pickle.dump(self.mattr_base_sorted, f)

    def _save_hapax_cdf(self) -> None:
        """
        Compute and save the hapax distributions of the static completions.

        :return: None
        """
        hapax_true = [
            self.calculate_hapax(
                c["messages"][2]["content"])
            for c in self.true_completions
        ]

        hapax_base = [
            self.calculate_hapax(
                c["messages"][2]["content"])
            for c in self.base_completions
        ]

        if not hapax_true:
            raise ValueError("No hapax legomena values computed. "
                             "True completions may be empty.")

        if not hapax_base:
            raise ValueError("No hapax legomena values computed. "
                             "Base completions may be empty.")

        self.hapax_true_sorted = np.sort(np.asarray(hapax_true, dtype=np.float64))
        self.hapax_base_sorted = np.sort(np.asarray(hapax_base, dtype=np.float64))

        hapax_file_true = DISTRIBUTIONS_DIR / "true" / HAPAX_FILENAME_TRUE
        hapax_file_base = DISTRIBUTIONS_DIR / "base" / HAPAX_FILENAME_BASE
        with hapax_file_true.open("wb") as f:
            pickle.dump(self.hapax_true_sorted, f)
        with hapax_file_base.open("wb") as f:
            pickle.dump(self.hapax_base_sorted, f)

    def fwf_true_cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the empirical CDF value(s) for function word frequency (true).

        :param x: 1D array of scalar function word frequency values.
        :return: 1D array of empirical CDF values in [0, 1].
        """
        if self.fwf_true_sorted is None:
            raise RuntimeError("FWF true distribution not loaded.")

        xs = np.asarray(x, dtype=np.float64)
        return np.searchsorted(
            self.fwf_true_sorted, xs, side="right"
        ) / len(self.fwf_true_sorted)

    def fwf_base_cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the empirical CDF value(s) for function word frequency (base).

        :param x: 1D array of scalar function word frequency values.
        :return: 1D array of empirical CDF values in [0, 1].
        """
        if self.fwf_base_sorted is None:
            raise RuntimeError("FWF base distribution not loaded.")

        xs = np.asarray(x, dtype=np.float64)
        return np.searchsorted(
            self.fwf_base_sorted, xs, side="right"
        ) / len(self.fwf_base_sorted)

    def mattr_true_cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the empirical CDF value(s) for MATTR (true).

        :param x: 1D array of scalar MATTR values.
        :return: 1D array of empirical CDF values in [0, 1].
        """
        if self.mattr_true_sorted is None:
            raise RuntimeError("MATTR true distribution not loaded.")

        xs = np.asarray(x, dtype=np.float64)
        return np.searchsorted(
            self.mattr_true_sorted, xs, side="right"
        ) / len(self.mattr_true_sorted)

    def mattr_base_cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the empirical CDF value(s) for MATTR (base).

        :param x: 1D array of scalar MATTR values.
        :return: 1D array of empirical CDF values in [0, 1].
        """
        if self.mattr_base_sorted is None:
            raise RuntimeError("MATTR base distribution not loaded.")

        xs = np.asarray(x, dtype=np.float64)
        return np.searchsorted(
            self.mattr_base_sorted, xs, side="right"
        ) / len(self.mattr_base_sorted)

    def hapax_true_cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the empirical CDF value(s) for hapax legomena frequency (true).

        :param x: 1D array of scalar hapax legomena frequency values.
        :return: 1D array of empirical CDF values in [0, 1].
        """
        if self.hapax_true_sorted is None:
            raise RuntimeError("Hapax true distribution not loaded.")

        xs = np.asarray(x, dtype=np.float64)
        return np.searchsorted(
            self.hapax_true_sorted, xs, side="right"
        ) / len(self.hapax_true_sorted)

    def hapax_base_cdf(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the empirical CDF value(s) for hapax legomena frequency (base).

        :param x: 1D array of scalar hapax legomena frequency values.
        :return: 1D array of empirical CDF values in [0, 1].
        """
        if self.hapax_base_sorted is None:
            raise RuntimeError("Hapax base distribution not loaded.")

        xs = np.asarray(x, dtype=np.float64)
        return np.searchsorted(
            self.hapax_base_sorted, xs, side="right"
        ) / len(self.hapax_base_sorted)

    @staticmethod
    def calculate_fwf(completion: str) -> float:
        """
        Compute the function word frequency of a completion.

        :param completion: a plain text completion of a prompt.
        :return: function word frequency of the completion.
        """
        stop_words = set(STOP_WORDS)

        words = re.findall(r"\b\w+\b", completion.lower())

        words_no_punct = [word for word in words if word not in string.punctuation]

        fw_count = sum(1 for word in words_no_punct if word in stop_words)

        return fw_count / len(words_no_punct) if words_no_punct else 0

    @staticmethod
    def calculate_mattr(
            completion: str,
            window: int = 100,
    ) -> float:
        """
        Compute the MATTR of a completion.

        :param completion: a plain text completion of a prompt.
        :param window: window size of the window used for computing MATTR.
        :return: MATTR of the completion.
        """
        words = re.findall(r"\b\w+\b", completion.lower())

        words_no_punct = [word for word in words if word not in string.punctuation]

        unique_words = set(words_no_punct)

        ttr = len(unique_words) / len(words) if words else 0

        # First ensure window is not larger than the number of words
        if len(words_no_punct) < window:
            mattr = ttr
        else:
            mattr = np.mean(
                [
                    len(set(words_no_punct[i : i + window])) / window
                    for i in range(0, len(words_no_punct), window)
                ]
            )

        return float(mattr)

    @staticmethod
    def calculate_hapax(
            completion: str,
    ) -> float:
        """
        Compute the hapax legomena frequency of a completion.

        :param completion: a plain text completion of a prompt.
        :return: hapax legomena frequency of the completion.
        """
        words = re.findall(r"\b\w+\b", completion.lower())

        words_no_punct = [word for word in words if word not in string.punctuation]

        hapax = sum(v==1 for v in Counter(words_no_punct).values())

        return hapax / len(words_no_punct) if words_no_punct else 0
