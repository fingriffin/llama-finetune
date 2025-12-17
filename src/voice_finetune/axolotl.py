"""Finetuning class for finetuning with Axolotl."""

import os
import subprocess
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from loguru import logger
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from voice_finetune.config import (
    FinetuneConfig,
    is_wandb_artifact,
    load_config_from_wandb_artifact,
    load_finetune_config,
)
from voice_finetune.hf import configure_hf, get_token


class Finetuner:
    """Wrapper for Axolotl CLI engine."""

    def __init__(
        self,
        config_path: str,
        wandb_run_id: str | None = None
    ) -> None:
        """
        Initialise the Finetuner with the given configuration.

        :param config_path: path to FinetuneConfig YAML file or wandb artifact
        :param wandb_run_id: optional wandb run ID to attach to
        :return: None
        """
        self.config_path: str = config_path
        self.local_config_path: Path | None = None
        self.wandb_run_id: str | None = wandb_run_id

        self.tokenizer: AutoTokenizer | None = None
        self.tokenizer_dir: str | None = None

        self.config: FinetuneConfig | None = None

        self._prepare_config()
        self._prepare_tokenizer()

    def train(self) -> None:
        """Start the finetuning process using Axolotl CLI."""
        if not self.local_config_path:
            raise ValueError("axolotl_config_path must be set before training.")

        env = os.environ.copy()

        # Inject wandb variables only if resuming a run
        if self.wandb_run_id:
            env["WANDB_RESUME"] = "must"
            env["WANDB_RUN_ID"] = self.wandb_run_id

            if "WANDB_PROJECT" in os.environ and self.config:
                env["WANDB_PROJECT"] = self.config.wandb_project

            if "WANDB_ENTITY" in os.environ and self.config:
                env["WANDB_ENTITY"] = self.config.wandb_entity

        subprocess.run(
            ["axolotl", "train", self.local_config_path],
            check=True,
            env=env,
        )

    def merge_and_push(self) -> None:
        """
        Merge the adapter and push to HF hub.

        :return: None
        """
        if self.config:

            logger.info("Downloading adapter repo from HF: {}", self.config.hub_model_id)
            repo_path = snapshot_download(repo_id=self.config.hub_model_id)
            adapter_path = repo_path # TODO: support for merging from earlier checkpoints

            from_pretrained_kwargs = {
                "torch_dtype": "bfloat16",
                "device_map": {"": 0},
            }

            logger.info(
                "Loading base model in full precision: {}",
                self.config.base_model
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                **from_pretrained_kwargs,
            )

            new_vocab_size = len(self.tokenizer) # type: ignore[arg-type]
            current_vocab_size = base_model.get_input_embeddings().weight.shape[0]
            if new_vocab_size != current_vocab_size:
                logger.info(
                    "Resizing token embeddings from {} to {}",
                    current_vocab_size,
                    new_vocab_size,
                )
                base_model.resize_token_embeddings(new_vocab_size, mean_resizing=False)
                base_model.config.vocab_size = new_vocab_size

            logger.info("Loading LoRA adapter from {}", adapter_path)
            peft_model = PeftModel.from_pretrained(base_model, adapter_path)

            logger.info("Merging LoRA adapter into base model weights...")
            merged_model = peft_model.merge_and_unload()

            # Save merged model into local directory
            model_dir = os.path.join(self.config.output_dir, "merged")
            os.makedirs(model_dir, exist_ok=True)

            logger.info("Saving merged model to {}", model_dir)
            merged_model.save_pretrained(model_dir, safe_serialization=True)
            self.tokenizer.save_pretrained(model_dir) # type: ignore[union-attr]

            # Push merged model to HF hub
            if self.config:
                merged_repo = f"{self.config.hub_model_id}-Merged"
                logger.info("Pushing merged model to HF Hub at {}", merged_repo)

            api = HfApi()
            api.create_repo(merged_repo, repo_type="model", exist_ok=True, private=True)
            api.upload_folder(
                folder_path=model_dir,
                repo_id=merged_repo,
                repo_type="model",
            )

            logger.success("Successfully pushed merged model to {}", merged_repo)

    def _prepare_config(self) -> None:
        """
        Prepare Axolotl configuration from the given YAML file.

        :return: None
        """
        if is_wandb_artifact(self.config_path):
            logger.info("Detected wandb artifact: {}", self.config_path)
            self.local_config_path = load_config_from_wandb_artifact(self.config_path)
            logger.info("Downloaded config to {}", str(self.local_config_path))
        else:
            self.local_config_path = Path(self.config_path).expanduser()

        try:
            logger.info("Loading config from {}", str(self.local_config_path))
            self.config = load_finetune_config(str(self.local_config_path))
            logger.success("Config loaded successfully!")
            print("Current configuration:")
            print(self.config.model_dump_json(indent=2))
            print("")
        except Exception as e:
            logger.error("Failed to load config: {}", e)
            raise

        configure_hf(self.config.base_model)
        get_token()

    def _prepare_tokenizer(self) -> None:
        """
        Prepare tokenizer for merge and push operation.

        :return: None
        """
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model, # type: ignore[union-attr]
            trust_remote_code=True,
            use_fast=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        self.tokenizer_dir = os.path.join(
            self.config.output_dir, # type: ignore[union-attr]
            "tokenizer"
        )
        os.makedirs(self.tokenizer_dir, exist_ok=True)
        self.tokenizer.save_pretrained(self.tokenizer_dir)
