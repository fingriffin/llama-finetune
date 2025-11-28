"""Utilities for finetuning with axolotl and experiment tracking with wandb."""

import os

import wandb
from dotenv import load_dotenv
from loguru import logger

# List of artifacts to keep when cleaning up wandb runs
KEEP_ARTIFACTS = [
    'MasterConfig',
    'FinetuneConfig',
    'InferenceConfig',
    'AnalyzeConfig',
    'Results',
    'BaseResults',
    'Analysis',
    'Log',
]

# List of config keys to keep when cleaning up wandb runs
KEEP_CONFIG_KEYS = [
    '0 base_model',
    '0 data_path',
    '1 finetune',
    '2 inference',
    '3 analyze',
]

def clean_wandb_run(wandb_run_id: str) -> None:
    """
    Clean artifacts form a wandb run.

    All are deleted unless they appear in KEEP_ARTIFACTS.

    :param wandb_run_id: wandb run id
    """
    api = wandb.Api()

    load_dotenv()
    entity = os.getenv("WANDB_ENTITY")
    project = os.getenv("WANDB_PROJECT")

    run_path = f"{entity}/{project}/{wandb_run_id}"
    run = api.run(run_path)

    logger.info("Cleaning artifacts for run", {run_path})

    for artifact in run.logged_artifacts():
        keep = (
            artifact.type in KEEP_ARTIFACTS
            or artifact.name in KEEP_ARTIFACTS
        )

        if keep:
            continue

        logger.info(f"Deleting artifact: {artifact.name} (type={artifact.type})")
        artifact.delete(delete_aliases=True)

    logger.info(f"Cleaning wandb config for {run_path}")

    # wandb config behaves like a dict
    current_keys = list(run.config.keys())

    for key in current_keys:
        if key not in KEEP_CONFIG_KEYS:
            logger.info(f"Deleting config key: {key}")
            del run.config[key]

    # Push the updated config to wandb server
    run.update()
