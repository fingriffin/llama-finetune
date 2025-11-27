"""Finetuning class for finetuning with Axolotl."""

import os
import subprocess
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import torch
import yaml
from axolotl.cli.config import load_cfg
from axolotl.utils.dict import DictDefault
from loguru import logger
from omegaconf import DictConfig, ListConfig
from transformers import AutoTokenizer

from voice_finetune.config import (
    FinetuneConfig,
    is_wandb_artifact,
    load_config_from_wandb_artifact,
    load_finetune_config,
)
from voice_finetune.hf import configure_hf, get_token


def to_plain(obj: Any) -> Any:
    """Convert Axolotl / OmegaConf objects into plain YAML-serialisable Python types."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, torch.dtype):
        return str(obj).replace("torch.", "")
    if isinstance(obj, DictConfig):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, ListConfig):
        return [to_plain(v) for v in obj]
    if isinstance(obj, DictDefault) or isinstance(obj, dict):
        return {k: to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    return obj


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
        self.wandb_run_id: str | None = wandb_run_id

        self.config: FinetuneConfig | None = None
        self.axolotl_config: DictDefault | None = None
        self.axolotl_config_path: str | None = None
        self._axolotl_whitelist_keys: set[str] | None = None

        self._prepare_configs()
        self._save_axolotl_config()

    def train(self, preprocess: bool = True) -> None:
        """
        Start the finetuning process using Axolotl CLI.

        :param preprocess: Whether to preprocess the data.
        :return: None
        """
        if self.axolotl_config_path:
            if preprocess:
                subprocess.run(
                    ["axolotl", "preprocess", self.axolotl_config_path],
                    check=True
                )
            subprocess.run(
                ["axolotl", "train", self.axolotl_config_path],
                check=True
            )

    def _prepare_configs(self) -> None:
        """
        Prepare FinetuneConfig and Axolotl configuration from the given YAML file.

        :return: None
        """
        if is_wandb_artifact(self.config_path):
            logger.info("Detected wandb artifact: {}", self.config_path)
            config_file = load_config_from_wandb_artifact(self.config_path)
            logger.info("Downloaded config to {}", str(config_file))
        else:
            config_file = Path(self.config_path).expanduser()

        try:
            logger.info("Loading config from {}", str(config_file))
            self.config = load_finetune_config(str(config_file))
            logger.success("Config loaded successfully!")
            print("Current configuration:")
            print(self.config.model_dump_json(indent=2))
            print("")
        except Exception as e:
            logger.error("Failed to load config: {}", e)
            raise

        configure_hf(self.config.model_name)
        get_token()

        if self.config.push_to_hub:
            model_name = os.path.basename(self.config.output_dir.rstrip("/"))
            hub_model_id = f"{os.getenv('HF_ORG')}/{model_name}"
            if self.config.checkpointing:
                hub_strategy = "every_save"
            else:
                hub_strategy = "end"
            logger.info("Will push adapter to the Hub with model ID: {}", hub_model_id)
        else:
            hub_model_id = None
            hub_strategy = None

        hf_org = os.getenv("HF_ORG")
        if hf_org:
            if str(self.config.train_data_path).startswith(hf_org + '/'):
                data_path = self.config.train_data_path
                logger.info(
                    f"Detected HF dataset: {str(self.config.train_data_path)}"
                )
        else:
            data_path = Path(self.config.train_data_path).expanduser().resolve()  # type: ignore[assignment]
            if not data_path.exists():  # type: ignore[attr-defined]
                logger.error(
                    "Training data not found at: {}",
                    str(data_path)
                )
                raise

        if self.config.checkpointing:
            logger.info(
                "Checkpointing enabled: will save model at the end of each epoch."
            )
            save_strategy = "epoch"
            save_total_limit = self.config.epochs
            save_only_model = False
        else:
            logger.info("Checkpointing disabled: will only save final model.")
            save_steps = 0
            save_strategy = "no"
            save_total_limit = 0
            save_only_model = True

        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer_dir = os.path.join(self.config.output_dir, "tokenizer")
        os.makedirs(tokenizer_dir, exist_ok=True)
        tokenizer.save_pretrained(tokenizer_dir)

        axolotl_cfg_raw = DictDefault(
            base_model=self.config.model_name,
            seed=self.config.seed,
            output_dir=self.config.output_dir,
            device_map=self.config.device_map,

            adapter=self.config.adapter,
            load_in_8bit=self.config.load_in_8bit,
            load_in_4bit=self.config.load_in_4bit,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            optimizer=self.config.optimizer,
            num_epochs=self.config.epochs,
            learning_rate=self.config.learning_rate,
            micro_batch_size=self.config.micro_batch_size,
            sequence_len=self.config.sequence_len,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            gradient_checkpointing=self.config.gradient_checkpointing,
            flash_attention=self.config.flash_attention,

            lora_r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            lora_target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],

            tokenizer_config=tokenizer_dir,
            special_tokens={
                "pad_token": "<PAD>",
            },

            save_steps=locals().get('save_steps', 0),
            save_strategy=save_strategy,
            save_total_limit=save_total_limit,
            save_only_model=save_only_model,

            datasets=[
                {
                    "path": str(data_path),
                    "split": "train",
                    "type": "chat_template",
                    "field_messages": "messages",
                    "message_field_role": "from",
                    "message_field_content": "value",
                }
            ],

            **(
                {
                    "test_datasets": [{
                        "path": str(data_path),
                        "split": "validation",
                        "type": "chat_template",
                        "field_messages": "messages",
                        "message_field_role": "from",
                        "message_field_content": "value",
                    }]
                }
                if self.config.do_validation
                else {}
            ),
            eval_steps=1,

            use_wandb=True,
            wandb_project=os.getenv('WANDB_PROJECT'),
            wandb_entity=os.getenv('WANDB_ENTITY'),
            wandb_watch="checkpoint",
            wandb_log_model="checkpoint",
            hub_model_id=hub_model_id,
            hub_strategy=hub_strategy,
        )

        self._axolotl_whitelist_keys = set(axolotl_cfg_raw.keys())
        self.axolotl_config = load_cfg(axolotl_cfg_raw)

    def _save_axolotl_config(self) -> None:
        """
        Save the prepared Axolotl configuration to a temporary YAML file.

        :return: None
        """
        if self.axolotl_config is None:
            raise RuntimeError("Axolotl config has not been prepared.")

        plain_cfg = to_plain(self.axolotl_config)

        if self._axolotl_whitelist_keys is not None:
            filtered_cfg = {
                key: plain_cfg[key]
                for key in self._axolotl_whitelist_keys
                if key in plain_cfg
            }
        else:
            filtered_cfg = plain_cfg

        with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml", mode="w") as f:
            yaml.safe_dump(filtered_cfg, f, sort_keys=False)
            self.axolotl_config_path = f.name
