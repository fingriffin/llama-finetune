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

# Alpha hyperparameter for the stylometric reward function
ALPHA = 0.15

# Batch distribution regularisation:
# LAMBDA_* control how strongly we penalise mismatch between the batch level
# distribution of generated style vectors and the batch level distribution
# of reference style vectors (within the current GRPO rollout batch)
LAMBDA_BATCH_MEAN = 0.05
LAMBDA_BATCH_COV = 0.0
BATCH_PENALTY_CLIP = 0.5

# Length regularisation:
# Penalises deviation of generated length from reference length (in words).
LAMBDA_LEN = 0.05
LEN_CLIP = 1.0

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

    gen_lengths = np.asarray(
        [manager._word_count(c[0]["content"]) for c in completions], dtype=np.float32
    )
    ref_lengths = np.asarray(
        [manager._word_count(t) for t in true_completions], dtype=np.float32
    )

    len_err = np.abs(gen_lengths - ref_lengths) / np.maximum(ref_lengths, 1.0)

    len_pen = np.clip(len_err, 0.0, LEN_CLIP)

    true_style_vectors = [
        manager.calculate_style_vector(c, n_words=N_WORDS)
        for c in true_completions
    ]

    # L1 distances (normed)
    distances = [
        manager.mahalanobis_distance(style_vectors[i], true_style_vectors[i])
        for i in range(len(style_vectors))
    ]

    # Batch-level distribution penalty
    X = np.stack(style_vectors)
    Y = np.stack(true_style_vectors)

    mu_X = X.mean(axis=0)
    mu_Y = Y.mean(axis=0)
    delta_mu = mu_X - mu_Y

    # Penalise mismatch between batch means, weighted by reference inverse covariance
    mean_pen = float(delta_mu.T @ manager.inv_cov_style @ delta_mu)

    cov_pen = 0.0
    if LAMBDA_BATCH_COV > 0.0 and X.shape[0] >= 2:
        cov_X = np.cov(X, rowvar=False)
        cov_Y = np.cov(Y, rowvar=False)
        cov_pen = float(np.sum((cov_X - cov_Y) ** 2))

    batch_pen = LAMBDA_BATCH_MEAN * mean_pen + LAMBDA_BATCH_COV * cov_pen
    batch_pen = float(min(batch_pen, BATCH_PENALTY_CLIP))

    base_rewards = np.asarray(
        [2.0 * np.exp(-ALPHA * d) - 1.0 for d in distances],
        dtype=np.float32
    )
    rewards = base_rewards - batch_pen - (LAMBDA_LEN * len_pen)
    rewards = np.clip(rewards, -1.0, 1.0)
    return rewards


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
