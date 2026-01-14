"""Finetune a model based on a given config file."""

import click

from voice_finetune.custom_logging import setup_logging
from voice_finetune.finetuner import Finetuner
from voice_finetune.utils import clean_wandb_run


@click.command()
@click.argument("config_path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
@click.option("--wandb-run-id", help="Attach to existing wandb run ID")
def main(
    config_path: str,
    log_level: str,
    log_file: str | None,
    wandb_run_id: str | None,
) -> None:
    """
    Run finetuning job based on the provided config file.

    :param config_path: Path to the config file.
    :param log_level: Optional override for logging level.
    :param log_file: Optional override for log file path.
    :param wandb_run_id: Optional wandb run ID to attach to.
    :return: None

    :raises Exception: If loading the finetune config fails.
    """
    # Setup logging
    setup_logging(level=log_level, log_file=log_file)

    # Launch axolotl engine
    finetuner = Finetuner(config_path=config_path, wandb_run_id=wandb_run_id)

    if finetuner.config and finetuner.config.vllm:
        finetuner.setup_vllm()

    # Train the model
    finetuner.train()

    # Merge and push to HF hub
    finetuner.merge_and_push()

    # Clean up wandb artifacts if specified
    if wandb_run_id:
        clean_wandb_run(wandb_run_id=wandb_run_id)
