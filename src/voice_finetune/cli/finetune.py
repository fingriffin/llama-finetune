"""Finetune a model based on a given config file."""

import click

from voice_finetune.axolotl import Finetuner
from voice_finetune.logging import setup_logging


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

    # Train the model
    finetuner.train()

    # # Merge and push LoRA adapters if specified
    # if config.do_merge and config.push_to_hub:
    #     logger.info("Downloading adapter repo from HF: {}", hub_model_id)
    #     repo_path = snapshot_download(repo_id=hub_model_id)
    #     adapter_path = os.path.join(repo_path, config.adapter_subfolder)
    #
    #     from_pretrained_kwargs = {
    #         "torch_dtype": "bfloat16",
    #         "device_map": {"": 0},
    #     }
    #
    #     logger.info("Loading base model in full precision: {}", config.model_name)
    #     base_model = AutoModelForCausalLM.from_pretrained(
    #         config.model_name,
    #         **from_pretrained_kwargs,
    #     )
    #
    #     new_vocab_size = len(tokenizer)
    #     current_vocab_size = base_model.get_input_embeddings().weight.shape[0]
    #     if new_vocab_size != current_vocab_size:
    #         logger.info(
    #             "Resizing token embeddings from {} to {}",
    #             current_vocab_size,
    #             new_vocab_size,
    #         )
    #         base_model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
    #         base_model.config.vocab_size = new_vocab_size
    #
    #     logger.info("Loading LoRA adapter from {}", adapter_path)
    #     peft_model = PeftModel.from_pretrained(base_model, adapter_path)
    #
    #     logger.info("Merging LoRA adapter into base model weights...")
    #     merged_model = peft_model.merge_and_unload()
    #
    #     # Save merged model into local directory
    #     model_dir = os.path.join(config.output_dir, "merged")
    #     os.makedirs(model_dir, exist_ok=True)
    #
    #     logger.info("Saving merged model to {}", model_dir)
    #     merged_model.save_pretrained(model_dir, safe_serialization=True)
    #     tokenizer.save_pretrained(model_dir)
    #
    #     # Push merged model to HF hub
    #     merged_repo = f"{hub_model_id}-Merged"
    #     logger.info("Pushing merged model to HF Hub at {}", merged_repo)
    #
    #     api = HfApi()
    #     api.create_repo(merged_repo, repo_type="model", exist_ok=True, private=False)
    #     api.upload_folder(
    #         folder_path=model_dir,
    #         repo_id=merged_repo,
    #         repo_type="model",
    #     )
    #
    #     logger.success("Successfully pushed merged model to {}", merged_repo)

