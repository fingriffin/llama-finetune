"""Reward functions for GRPO fine-tuning."""

from typing import Any

import numpy as np

from voice_finetune.distributions import DistributionManager


def fwf_reward_func(
        completions: list[list[dict[str,str]]],
        **kwargs: dict
) -> Any:
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

    fwf_kde_base = manager.fwf_kde_base

    fwf_kde_true = manager.fwf_kde_true

    if not fwf_kde_true or not fwf_kde_base:
        raise RuntimeError("Failed to load FWF KDEs.")

    fwfs = np.asarray(
        [
            manager.calculate_fwf(c[0]["content"])for c in completions
        ]
    )

    return np.tanh(
        np.log(fwf_kde_true(fwfs)) - np.log(fwf_kde_base(fwfs))
    ).tolist()

def mattr_reward_func(
        completions: list[list[dict[str,str]]],
        **kwargs: dict
) -> Any:
    """
    Return the MATTR reward for each completion.

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
    :return: List of MATTR reward values.
    """
    _ = kwargs

    manager = DistributionManager(mattr=True)

    mattr_kde_base = manager.mattr_kde_base

    mattr_kde_true = manager.mattr_kde_true

    if not mattr_kde_base or not mattr_kde_true:
        raise RuntimeError("Failed to load MATTR KDE.")

    mattrs = np.asarray(
        [
            manager.calculate_mattr(c[0]["content"])for c in completions
        ]
    ).tolist()

    return np.tanh(
        np.log(mattr_kde_true(mattrs)) - np.log(mattr_kde_base(mattrs))
    ).tolist()
