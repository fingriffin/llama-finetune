"""Reward functions for GRPO fine-tuning."""

from typing import Any

import numpy as np

from voice_finetune.reward_manager import RewardManager

# Defaults (can be overridden via function args)
N_WORDS = 200
ALPHA = 0.15

LAMBDA_BATCH_MEAN = 0.05
LAMBDA_BATCH_VAR = 0.05
BATCH_PENALTY_CLIP = 0.5

LAMBDA_LEN = 0.05
LEN_CLIP = 1.0


def stylometric_reward_func(
        prompts: list[list[dict[str, str]]],
        completions: list[list[dict[str, str]]],
        *,
        n_words: int = N_WORDS,
        alpha: float = ALPHA,
        lambda_batch_mean: float = LAMBDA_BATCH_MEAN,
        lambda_batch_var: float = LAMBDA_BATCH_VAR,
        batch_penalty_clip: float = BATCH_PENALTY_CLIP,
        lambda_len: float = LAMBDA_LEN,
        len_clip: float = LEN_CLIP,
        **kwargs: dict
) -> Any:
    """
    Return the stylometric reward for each completion.

    Per sample base reward uses squared Mahalanobis distance d between style vectors:
        r_i = 2 * exp(-alpha * d_i) - 1

    Batch regularisation penalises mismatch of batch mean and per-dimension variance
    between generated and reference style vectors.

    Length regularisation penalises relative deviation of generated length from
    reference length (in words).

    The parameter prompts has the following structure:

    prompts = [
        [ {"role": "user", "content": "..."}, ],
        ...
    ]

    The parameter completions has the following structure:

    completions = [
        [ {"role": "assistant", "content": "..."}, ],
        ...
    ]

    :param prompts: List of prompts.
    :param completions: List of completions.
    :param n_words: Number of words to compute style features on (prefix length).
    :param alpha: Exponential decay for per-sample style mismatch (larger = stricter).
    :param lambda_batch_mean: Strength of batch mean-matching penalty.
    :param lambda_batch_var: Strength of batch variance matching penalty (diag only).
    :param batch_penalty_clip: Clip applied to total batch penalty for stability.
    :param lambda_len: Strength of per-sample length penalty.
    :param len_clip: Clip applied to per-sample relative length error.
    :param kwargs: Keyword arguments from trainer.
    :return: List/array of stylometric reward values.
    """
    _ = kwargs

    manager = RewardManager()

    # Generated + reference texts
    gen_texts = [c[0]["content"] for c in completions]
    ref_texts = [
        manager.get_true_completion(next(m["content"] for m in p if m["role"] == "user"))
        for p in prompts
    ]

    # Style vectors
    X_list = [manager.calculate_style_vector(t, n_words=n_words) for t in gen_texts]
    Y_list = [manager.calculate_style_vector(t, n_words=n_words) for t in ref_texts]

    # Per-sample distances and base reward
    d = np.asarray(
        [manager.mahalanobis_distance(X_list[i], Y_list[i]) for i in range(len(X_list))],
        dtype=np.float32,
    )
    base_rewards = 2.0 * np.exp(-alpha * d) - 1.0

    # Per-sample length penalty (relative word count mismatch)
    gen_len = np.asarray([manager._word_count(t) for t in gen_texts], dtype=np.float32)
    ref_len = np.asarray([manager._word_count(t) for t in ref_texts], dtype=np.float32)
    len_err = np.abs(gen_len - ref_len) / np.maximum(ref_len, 1.0)
    len_pen = np.clip(len_err, 0.0, len_clip)

    # Batch penalties (mean + diagonal variance)
    X = np.stack(X_list)
    Y = np.stack(Y_list)

    mean_pen = 0.0
    if lambda_batch_mean > 0.0:
        delta_mu = X.mean(axis=0) - Y.mean(axis=0)
        mean_pen = float(delta_mu.T @ manager.inv_cov_style @ delta_mu)

    var_pen = 0.0
    if lambda_batch_var > 0.0 and X.shape[0] >= 2:
        var_pen = float(np.sum((X.var(axis=0, ddof=1) - Y.var(axis=0, ddof=1)) ** 2))

    batch_pen = lambda_batch_mean * mean_pen + lambda_batch_var * var_pen
    batch_pen = float(min(batch_pen, batch_penalty_clip))

    rewards = base_rewards - batch_pen - (lambda_len * len_pen)
    rewards = np.clip(rewards, -1.0, 1.0)
    return rewards
