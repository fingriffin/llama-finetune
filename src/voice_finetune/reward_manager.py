"""Module for fast true completion lookups and reward calculation utils."""

import json
import re
import string
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np
from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TRAIN_FILENAME = "train.jsonl"
VALIDATION_FILENAME = "validation.jsonl"

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

class RewardManager:
    """Class for fast true completion lookups and reward calculation utils."""

    def __init__(
            self
        ) -> None:
        """
        Initialise RewardManger.

        Loads training data from data directory into true_completions.

        :return: None
        """
        # Completions of prompts/questions from true corpus
        # This is loaded from the training set
        self.true_completions: List[dict] = []
        # Used to z-score style vectors
        self.mean_style_vector: np.ndarray = np.array([])
        self.std_style_vector: np.ndarray = np.array([])

        DATA_DIR.mkdir(exist_ok=True)
        self._load_true_completions()
        self._calculate_mean_and_std()

    def _load_true_completions(self) -> None:
        """
        Load true completions from jsonl files.

        If the files do not exist, create them first.

        :return: None
        """
        train_file = DATA_DIR / TRAIN_FILENAME
        val_file = DATA_DIR / VALIDATION_FILENAME
        if not (train_file.exists() and val_file.exists()):
            self._save_true_completions()

        with train_file.open("r", encoding="utf-8") as f:
            true_completions_train = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        with val_file.open("r", encoding="utf-8") as f:
            true_completions_val = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

        self.true_completions = true_completions_train + true_completions_val

    def _calculate_mean_and_std(self) -> None:
        """
        Calculate vector of mean and std deviation in each style dimension.

        :return: None
        """
        if not self.true_completions:
            raise RuntimeError("Attribute true_completions empty.")

        vectors = []
        for c in self.true_completions:
            style_vector = self.calculate_style_vector(
                next(
                    msg["content"] for msg in c["messages"] if msg["role"] == "assistant"
                )
            )
            vectors.append(style_vector)

        self.mean_style_vector = np.mean(vectors, axis=0)
        self.std_style_vector = np.std(vectors, axis=0)

        # Floor std after computing
        self.std_style_vector = np.maximum(self.std_style_vector, 1e-3)

    @staticmethod
    def _save_true_completions() -> None:
        """
        Save the true completions to disk.

        Saves both training and validation set.

        :return: None
        """
        ds_train = load_dataset(
            "AccelerateScience/bush-dataset", # TODO: Config controlled
            split="train"
        )

        ds_val = load_dataset(
            "AccelerateScience/bush-dataset", # TODO: Config controlled
            split="validation"
        )

        ds_train.to_json(DATA_DIR / TRAIN_FILENAME)
        ds_val.to_json(DATA_DIR / VALIDATION_FILENAME)

    def get_true_completion(self, prompt: str) -> str:
        """
        Get true completion for a given prompt from the training data.

        :param prompt: prompt to get the true completion for
        :return: true completion for given prompt
        """
        if not self.true_completions:
            raise RuntimeError("Attribute true_completions empty.")

        # Build caches once (lazy init)
        if not hasattr(self, "_true_completion_exact"):
            self._true_completion_exact: dict[str, str] = {}
            self._true_completion_prefix: dict[str, str] = {}
            self._true_completion_prefix_n: int = 12

            for item in self.true_completions:
                msgs = item["messages"]
                p = msgs[1]["content"].strip()
                c = msgs[2]["content"]

                self._true_completion_exact[p] = c

                key = " ".join(p.split()[:self._true_completion_prefix_n])
                self._true_completion_prefix[key] = p

        prompt = prompt.strip()

        # Exact match (fast path)
        if prompt in self._true_completion_exact:
            return self._true_completion_exact[prompt]

        # Prefix match fallback
        key = " ".join(prompt.split()[:self._true_completion_prefix_n])
        if key in self._true_completion_prefix:
            full_prompt = self._true_completion_prefix[key]
            return self._true_completion_exact[full_prompt]

        raise KeyError("No true completion found for given prompt.")

    def calculate_style_vector(
            self,
            completion: str,
            *,
            z_score: bool = False,
    ) -> np.ndarray:
        """
        Calculate style vector of a given completion.

        The style vector is a 3-dimensional vector with elements:
        - Function/stop word frequency (FWF)
        - Moving average token type ratio (MATTR)
        - Hapax legomena frequency

        :param completion: completion to calculate style vector for
        :param z_score: whether to z score wrt true completions
        :return: style vector (3-dimensional)
        """
        fwf = self._calculate_fwf(completion)
        mattr = self._calculate_mattr(completion)
        hapax = self._calculate_hapax(completion)

        style_vector = np.array(
            [
                fwf,
                mattr,
                hapax,
            ]
        )

        if z_score:
            return (style_vector - self.mean_style_vector) / self.std_style_vector
        else:
            return style_vector

    @staticmethod
    def _calculate_fwf(completion: str) -> float:
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
    def _calculate_mattr(
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
        words_no_punct = [w for w in words if w not in string.punctuation]

        n = len(words_no_punct)
        if n == 0:
            return 0.0

        # If shorter than window, MATTR degenerates to TTR on full text
        if n < window:
            return float(len(set(words_no_punct)) / n)

        ttrs = []
        for i in range(0, n - window + 1):  # stride 1
            w = words_no_punct[i: i + window]
            ttrs.append(len(set(w)) / window)

        return float(np.mean(ttrs))

    @staticmethod
    def _calculate_hapax(
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
