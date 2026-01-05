"""Reward functions for GRPO fine-tuning."""

import random
from typing import Any

from datasets import load_dataset

from voice_finetune.hf import get_token

BASE_COMPLETIONS_CACHE: list[Any] = []
DATASET = "AccelerateScience/bush-dataset" # TODO: Remove hardcoding

def _get_base_completions() -> list[Any]:
    """
    Return and cache base completions for re-use in reward functions.

    :return: base completions
    """
    if BASE_COMPLETIONS_CACHE:
        return BASE_COMPLETIONS_CACHE

    get_token()

    dataset = load_dataset(DATASET, split="train")

    for example in dataset:
        example_dict = dict(example)
        messages = example_dict.get("messages", [])

        for msg in messages:
            if msg.get("role") == "assistant":
                BASE_COMPLETIONS_CACHE.append(msg.get("content", ""))
                break

    return BASE_COMPLETIONS_CACHE


# TODO: replace with real reward logic
def rand_reward_func(
        completions: list[list[dict[str,str]]],
        **kwargs: dict
) -> list[float]:
    """
    Return a random reward for each completion (placeholder).

    :param completions: list of completions from model
    :return: list of random rewards for each completion
    """
    _ = kwargs

    base_completions = _get_base_completions()

    # TODO: Remove below (temporary debugging)
    print(completions)
    print(base_completions)

    return [random.uniform(0, 1) for _ in completions]
