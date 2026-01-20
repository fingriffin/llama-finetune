"""Module for fast true completion lookups and reward calculation."""

import json
from pathlib import Path
from typing import List

from datasets import load_dataset

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
TRAIN_FILENAME = "train.jsonl"

class RewardManager:
    """Class for fast true completion lookups and reward calculation."""

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

        DATA_DIR.mkdir(exist_ok=True)
        self._load_true_completions()

    def _load_true_completions(self) -> None:
        """
        Load true completions from jsonl file.

        If the files do not exist, create them first.

        :return: None
        """
        file = DATA_DIR / TRAIN_FILENAME
        if not file.exists():
            self._save_true_completions()

        with file.open("r", encoding="utf-8") as f:
            self.true_completions = [
                json.loads(line)
                for line in f
                if line.strip()
            ]

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
        ds.to_json(DATA_DIR / TRAIN_FILENAME)

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

