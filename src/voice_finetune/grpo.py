"""Module containing GRPO RLHF utilities."""

from typing import Any, Callable

PROMPT_KEY = "prompt"
MESSAGES_KEY = "messages"
ASSISTANT_KEY = "assistant"

def grpo_transform(cfg: Any, *args: Any, **kwargs: Any) -> Callable[..., Any]:
    """
    Transform (patched) dataset for GRPO training.

    Prevents GRPO training calling DPO (preference-based) dataset processing logic.

    Ensures each example has a 'prompt' key for compatibility with
    trl.grpo_trainer._generate_and_score_completions.

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

        if MESSAGES_KEY not in example:
            raise KeyError(
                f"GRPO requires a `{MESSAGES_KEY}` field for chat datasets."
            )

        messages = example[MESSAGES_KEY]

        prompt_messages = [
            m for m in messages if m.get("role") != ASSISTANT_KEY
        ]

        if not prompt_messages:
            raise ValueError("Prompt is empty after removing assistant messages.")

        example[PROMPT_KEY] = prompt_messages

        return example

    return transform_fn
