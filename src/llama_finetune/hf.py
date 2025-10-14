"""Hugging Face configuration."""

import os
from dotenv import load_dotenv

def configure_hf(model_name: str):
    dir = f"models/{model_name}"
    os.environ["HF_HOME"] =  dir
    os.environ["TRANSFORMERS_CACHE"] = dir

def get_token():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found in .env file.")