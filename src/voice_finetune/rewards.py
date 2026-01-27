"""Reward functions for GRPO fine-tuning."""

from typing import Any

import numpy as np

from voice_finetune.distributions import DistributionManager
from voice_finetune.reward_manager import RewardManager

# Threshold CDF at which minimum reward is given.
# This refers to measurements against the base distribution
CDF_THRESHOLD = 0.01

# Number of words to calculate rewards on
N_WORDS = 200

ALPHA = 0.15

def stylometric_reward_func(
        prompts: list[list[dict[str,str]]],
        completions: list[list[dict[str,str]]],
        **kwargs: dict
) -> Any:
    """
    Return the stylometric reward for each completion.

    The reward depends on the L1 distance (d) between the
    style vectors of the generated and true completions:

    R = 2 * exp(-alpha*d) - 1

    The parameter prompts has the following structure:

    prompts = [
        [ {"role": "user", "content": "..."}, ],
        [ {"role": "user", "content": "..."}, ],
        [ {"role": "user", "content": "..."}, ],
        ... repeated num_generations times
    ]

    The parameter completions has the following structure:

    completions = [
        [ {"role": "assistant", "content": "..."}, ],
        [ {"role": "assistant", "content": "..."}, ],
        [ {"role": "assistant", "content": "..."}, ],
        ... repeated num_generations times
    ]

    :param prompts: List of prompts.
    :param completions: List of completions.
    :param kwargs: Keyword arguments from trainer.
    :return: List of stylometric reward values.
    """
    _ = kwargs

    manager = RewardManager()

    style_vectors = [
        manager.calculate_style_vector(c[0]["content"], n_words=N_WORDS)
        for c in completions
    ]

    true_completions = [
        manager.get_true_completion(next(m["content"] for m in p if m["role"] == "user"))
        for p in prompts
    ]

    true_style_vectors = [
        manager.calculate_style_vector(c, n_words=N_WORDS)
        for c in true_completions
    ]

    # L1 distances (normed)
    distances = [
        manager.mahalanobis_distance(style_vectors[i], true_style_vectors[i])
        for i in range(len(style_vectors))
    ]

    return [
        2.0 * np.exp(-ALPHA * d) - 1.0
        for d in distances
    ]

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
