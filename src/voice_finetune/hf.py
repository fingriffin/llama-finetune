"""Hugging Face configuration."""

import os

from dotenv import load_dotenv


def configure_hf(model_name: str) -> None:
    """
    Configure Hugging Face cache directories based on model name.

    :param model_name: Name of the Hugging Face model.
    :return: None
    """
    dir = f"models/{model_name}"
    os.environ["HF_HOME"] = dir
    os.environ["TRANSFORMERS_CACHE"] = dir


def get_token() -> None:
    """
    Load Hugging Face token from .env file.

    :raises RuntimeError: If HF_TOKEN is not found in .env file.
    :return: None
    """
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found in .env file.")
