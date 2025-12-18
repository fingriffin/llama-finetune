"""Module containing GRPO RLHF utilities."""

from typing import Any, Callable


def grpo_transform(cfg: Any, *args: Any, **kwargs: Any) -> Callable[..., Any]:
    """
    Identity transformation to patch GRPO training.

    Prevents GRPO training calling DPO (preference-based) dataset processing logic.

    Known bug cf. https://github.com/axolotl-ai-cloud/axolotl/issues/2986

    :param cfg: _
    :param args: _
    :param kwargs: _
    :return: _
    """
    _ = cfg
    __ = args
    ___ = kwargs

    def transform_fn(example: Any, tokenizer: Any | None = None) -> Any:
        _ = tokenizer
        return example

    return transform_fn
