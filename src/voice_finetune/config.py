"""Configuration management for finetuning with Axolotl."""

import os
import shutil
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from wandb import Api


class TRLConfig(BaseModel):
    """Configuration for fine-tuning with TRL."""

    max_completion_len: int = Field(
        2048,
        description="Maximum token length of completions during TRL"
    )

    use_vllm: bool = Field(False, description="Whether to use vLLM during training")

    reward_funcs: list[str] = Field(..., description="List of stylometric rewards to use")
    reward_weights: list[float] = Field(
        ...,
        description="List of weights for stylometric rewards"
    )
    num_generations: int = Field(1, description="Number of generations to sample")
    log_completions: bool = Field(
        False,
        description="Whether to log completions during training"
    )

    # Hardcoded as None to force GRPO training
    sync_ref_model: bool | None = Field(
        None,
        description="Whether to synchronize the baseline policy during training"
    )
    ref_model_mixup_alpha: float | None = Field(
        None,
        description="The mixup alpha parameter for the reference model."
    )
    ref_model_sync_steps: int | None = Field(
        None,
        description="The number of steps to synchronize the reference model."
    )

    scale_rewards: bool = Field(
        False,
        description="Whether to scale rewards by std deviation during training"
    )

    temperature: float = Field(0.7, description="Temperature for the RL policy")
    top_p: float | None = Field(0.9, description="Top-p value for generation policy")
    top_k: int | None = Field(None, description="Top-k sampling for generation policy")
    num_iterations: int = Field(1, description="Number of iterations per batch for GRPO")

class FinetuneConfig(BaseModel):
    """Configuration for LoRA/QLoRA finetuning."""

    base_model: str = Field(..., description="Name of the model to use")
    seed: int = Field(42, description="Random seed")
    output_dir: str = Field(..., description="Directory to save checkpoints and outputs")
    device_map: str = Field("auto", description="Device map for model loading")

    adapter: str = Field(..., description="Name of the adapter model to use")
    load_in_8bit: bool = Field(False, description="Load the model from 8bit")
    load_in_4bit: bool = Field(False, description="Load the model from 4bit")
    bf16: bool = Field(False, description="Load the model from BF16")
    fp16: bool = Field(True, description="Load the model from FP16")
    optimizer: str = Field("paged_adamw_32bit", description="Optimizer to use")
    num_epochs: int = Field(3, description="Number of training epochs")
    learning_rate: float = Field(2e-4, description="Learning rate")
    micro_batch_size: int = Field(2, description="Batch size per device")
    sequence_len: int = Field(1024, description="Maximum sequence length")
    gradient_accumulation_steps: int = Field(4, description="No. of accumulation steps")
    gradient_checkpointing: bool = Field(False, description="Use gradient checkpointing")
    flash_attention: bool = Field(False, description="Use flash attention if available")

    lora_r: int = Field(8, description="LoRA rank")
    lora_alpha: int = Field(16, description="LoRA alpha")
    lora_dropout: float = Field(0.05, description="LoRA dropout")
    lora_target_modules: list[str] |  None = Field(
        None,
        description="List of target modules for LoRA",
    )

    rl: Optional[str] = Field(None, description="Name of RL model to use (e.g. GRPO)")
    trl: Optional[TRLConfig] = Field(
        None,
        description="Optional configuration for TRL"
    )

    tokenizer_config: str | None = Field(None, description="Tokenizer config")
    special_tokens: dict[str, str] | None = Field(None, description="Special tokens dict")

    save_steps: int | float | None = Field(
        0,
        description="When to save model checkpoints",
    )
    save_strategy: str | None = Field("no", description="Saving strategy")
    save_total_limit: int = Field(
        0,
        description="Maximum number of checkpoints to save at one point"
    )
    save_only_model: bool = Field(
        True,
        description="Whether to save only the model",
    )

    datasets: list[dict[str, str]] = Field(
        ...,
        description="Datasets to use"
    )
    test_datasets: list[dict[str, str]] = Field(
        ...,
        description="Validation datasets to use"
    )
    eval_steps: int | None = Field(
        1,
        description="How often to run validation, in steps"
    )

    use_wandb: bool = Field(True, description="Whether to use wandb")
    wandb_project: str = Field(
        os.getenv("WANDB_PROJECT"),
        description="wandb project name",
    )
    wandb_entity: str = Field(
        os.getenv("WANDB_ENTITY"),
        description="wandb entity name",
    )
    wandb_watch: str = Field(
        "checkpoint",
        description="When to log model artifact"
    )
    wandb_log_model: str = Field(
        "checkpoint",
        description="When to log model artifact"
    )
    hub_model_id: str = Field(
        ...,
        description="Where to push checkpoints to on HF hub"
    )
    hub_strategy: str = Field(
        "end",
        description="How to push checkpoints to HF hub"
    )

    @field_validator("output_dir")
    def create_output_path(cls, v: str) -> str:
        """Ensure output directory exists."""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

def load_finetune_config(config_path: str) -> FinetuneConfig:
    """
    Load and validate finetuning configuration from YAML file.

    :param config_path: Path to configuration file
    """
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return FinetuneConfig(**config_dict)

def is_wandb_artifact(uri: str) -> bool:
    """
    Check if the given URI points to a Weights & Biases artifact.

    :param uri: URI to check
    :return: True if URI is a W&B artifact, False otherwise
    """
    # Local file exists → not an artifact
    if Path(uri).expanduser().exists():
        return False

    # Must contain "/" and ":" in typical positions
    return ("/" in uri) and (":" in uri)

def load_config_from_wandb_artifact(uri: str) -> Path:
    """
    Load FinetuneConfig YAML file from wandb artifact.

    :param uri: wandb artifact URI i.e. entity/project/artifact:version
    :return: path to downloaded YAML file
    """
    load_dotenv()
    api = Api()

    artifact = api.artifact(uri, type=None)

    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(artifact.download())

    yaml_files = list(tmp_dir.glob("*.yaml")) + list(tmp_dir.glob("*.yml"))
    if not yaml_files:
        raise RuntimeError(f"No YAML files found inside artifact: {uri}")
    if len(yaml_files) > 1:
        raise RuntimeError(f"Multiple YAML files found inside artifact: {uri}")

    yaml_src = yaml_files[0]

    # Ensure local configs directory exists
    configs_dir = Path("configs")
    configs_dir.mkdir(parents=True, exist_ok=True)

    # Construct destination filename
    artifact_name = uri.split("/")[-1].split(":")[0]
    dest_path = configs_dir / f"{artifact_name}.yaml"

    # Save YAML file
    dest_path.write_text(yaml_src.read_text())

    # Delete the downloaded artifact directory
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Delete the wandb artifacts/ folder
    artifacts_dir = Path("artifacts")
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir, ignore_errors=True)

    return dest_path
