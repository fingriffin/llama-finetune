"""Reward functions for GRPO fine-tuning."""

import random


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

    return [random.uniform(0, 1) for _ in completions]
