"""Reward functions for GRPO fine-tuning."""

from typing import Any

import numpy as np

from voice_finetune.distributions import DistributionManager

# Threshold CDF at which minimum reward is given.
# This refers to measurements against the base distribution
CDF_THRESHOLD = 0.01

def stylometric_reward_func(
        completions: list[list[dict[str,str]]],
        **kwargs: dict
) -> Any:
    """
    Return the stylometric reward for each completion.

    The result is the average of rewards associated with each metric:
    - Function word frequency (FWF)
    - Moving average type-token ratio (MATTR)

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
    :return: List of stylometric reward values.
    """
    _ = kwargs

    fwf_rewards = fwf_reward_func(completions)
    mattr_rewards = mattr_reward_func(completions)
    hapax_rewards = hapax_reward_func(completions)

    return (fwf_rewards + mattr_rewards + hapax_rewards) / 3

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

    fwfs = np.asarray(
        [
            manager.calculate_fwf(c[0]["content"])for c in completions
        ]
    )

    u_true = manager.fwf_true_cdf(fwfs)
    u_base = manager.fwf_base_cdf(fwfs)

    # True centrality -> [-1, 1]
    c_true = 1.0 - 2.0 * np.abs(u_true - 0.5)
    result = 2.0 * c_true - 1.0

    # Forbid extreme OOD
    bad = (u_base < CDF_THRESHOLD) | (u_base > 1.0 - CDF_THRESHOLD)
    result[bad] = -1.0

    return result

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

    mattrs = np.asarray(
        [
            manager.calculate_mattr(c[0]["content"])for c in completions
        ]
    )

    u_true = manager.mattr_true_cdf(mattrs)
    u_base = manager.mattr_base_cdf(mattrs)

    # True centrality -> [-1, 1]
    c_true = 1.0 - 2.0 * np.abs(u_true - 0.5)
    result = 2.0 * c_true - 1.0

    # Forbid extreme OOD
    bad = (u_base < CDF_THRESHOLD) | (u_base > 1.0 - CDF_THRESHOLD)
    result[bad] = -1.0

    return result

def hapax_reward_func(
        completions: list[list[dict[str,str]]],
        **kwargs: dict
) -> Any:
    """
    Return the hapax legomena reward for each completion.

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
    :return: List of hapax legomena reward values.
    """
    _ = kwargs

    manager = DistributionManager(hapax=True)

    hapaxs = np.asarray(
        [
            manager.calculate_hapax(c[0]["content"])for c in completions
        ]
    )

    u_true = manager.hapax_true_cdf(hapaxs)
    u_base = manager.hapax_base_cdf(hapaxs)

    # True centrality -> [-1, 1]
    c_true = 1.0 - 2.0 * np.abs(u_true - 0.5)
    result = 2.0 * c_true - 1.0

    # Forbid extreme OOD
    bad = (u_base < CDF_THRESHOLD) | (u_base > 1.0 - CDF_THRESHOLD)
    result[bad] = -1.0

    return result
