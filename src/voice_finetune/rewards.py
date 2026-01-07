"""Reward functions for GRPO fine-tuning."""

import numpy as np

from voice_finetune.distributions import DistributionManager


def fwf_reward_func(
        completions: list[list[dict[str,str]]],
        **kwargs: dict
) -> list[float]:
    """
    Return the function word frequency reward for each completion.

    The parameter completions has the following structure:

    [
        [
            {"role": "assistant", "content": ...},
            {"role": "assistant", "content": ...},
            ... continued num_generations times,
        ]
    ]

    :param completions: List of completions.
    :param kwargs: Keyword arguments from trainer.
    :return: List of fwf reward values.
    """
    _ = kwargs

    manager = DistributionManager(fwf=True)

    fwf_kde = manager.fwf_kde

    if not fwf_kde:
        raise RuntimeError("Failed to load FWF KDE.")

    fwfs = [
        manager.calculate_fwf(c[0]["content"])
        for c in completions
    ]

    return np.log(fwf_kde(fwfs)).tolist()
