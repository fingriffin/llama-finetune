"""Hugging Face configuration."""

import os
from dotenv import load_dotenv

def configure_hf():
    os.environ["HF_HOME"] =  "models/"
    os.environ["TRANSFORMERS_CACHE"] = "models/"

def get_token():
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise RuntimeError("HF_TOKEN not found in .env file.")