"""Module to manage persistent storage of stylometric distributions."""

import json
import pickle
import re
import string
from pathlib import Path
from typing import List

import numpy as np
from datasets import load_dataset
from scipy.stats import gaussian_kde

ROOT_DIR = Path(__file__).resolve().parents[2]
DISTRIBUTIONS_DIR = ROOT_DIR / "data" / "distributions"
BASE_COMPLETIONS_FILENAME = "train.jsonl"
FWF_FILENAME = "fwf.pkl"
MATTR_FILENAME = "mattr.pkl"

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
        ) -> None:
        """
        Initialize the distribution manager.

        :param fwf: If True, import function word frequency distribution
        """
        self.fwf = fwf
        self.mattr = mattr

        self.base_completions: List[dict] = []
        self.fwf_kde = None
        self.mattr_kde = None

        DISTRIBUTIONS_DIR.mkdir(parents=True, exist_ok=True)
        self.files = self._list_distribution_files()
        self._load_base_completions()

        if self.fwf:
            self._load_or_create_fwf()

        if self.mattr:
            self._load_or_create_mattr()

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

    def _load_or_create_mattr(self) -> None:
        """
        Load MATTR (Moving average TTR) KDE if it exists.

        Otherwise, compute and save it.

        :return: None
        """
        mattr_file = DISTRIBUTIONS_DIR / FWF_FILENAME
        if mattr_file.exists():
            with mattr_file.open("rb") as f:
                self.mattr_kde = pickle.load(f)
        else:
            self._save_mattr_kde()

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

    def _save_mattr_kde(self) -> None:
        """
        Compute and save the MATTR distribution of the base completions.

        :return: None
        """
        mattrs = [
            self.calculate_mattr(
                c["messages"][2]["content"])
            for c in self.base_completions
        ]

        if not mattrs:
            raise ValueError("No MATTR values computed. "
                             "Base completions may be empty.")

        # Fit KDE to the scalar FWF values
        kde = gaussian_kde(np.array(mattrs))

        # Store in instance
        self.mattr_kde = kde

        # Save KDE to disk
        mattr_file = DISTRIBUTIONS_DIR / MATTR_FILENAME
        with mattr_file.open("wb") as f:
            pickle.dump(kde, f)

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
