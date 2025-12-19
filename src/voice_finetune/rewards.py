"""Reward functions for GRPO fine-tuning."""

import random


# TODO: replace with real reward logic
def rand_reward_func(prompts: list[str], completions: list[str]) -> list[float]:
    """
    Return a random reward for each completion (placeholder).

    :param prompts: list of prompts
    :param completions: list of completions from model
    :return: list of random rewards for each completion
    """
    _ = prompts
    return [random.uniform(0, 1) for _ in completions]
