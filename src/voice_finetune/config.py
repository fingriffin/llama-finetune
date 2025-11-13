"""Configuration management for finetuning with Axolotl."""

import shutil
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from wandb import Api


class FinetuneConfig(BaseModel):
    """Configuration for LoRA/QLoRA finetuning."""

    model_name: str = Field(..., description="Name of the model to use")
    adapter: str = Field(..., description="Name of the adpter model to use")
    train_data_path: str = Field(..., description="Path to training data")
    output_dir: str = Field(..., description="Directory to save checkpoints and outputs")

    load_in_8bit: bool = Field(False, description="Load the model from 8bit")
    load_in_4bit: bool = Field(False, description="Load the model from 4bit")
    bf16: bool = Field(False, description="Load the model from BF16")
    fp16: bool = Field(True, description="Load the model from FP16")
    gradient_checkpointing: bool = Field(True, description="Use gradient checkpointing")

    optimizer: str = Field("paged_adamw_32bit", description="Optimizer to use")
    gpus: int = Field(1, description="Number of GPUs to use")

    epochs: int = Field(3, description="Number of training epochs")
    micro_batch_size: int = Field(2, description="Batch size per device")
    gradient_accumulation_steps: int = Field(4, description="No. of accumulation steps")
    learning_rate: float = Field(2e-4, description="Learning rate")

    lora_r: int = Field(8, description="LoRA rank")
    lora_alpha: int = Field(16, description="LoRA alpha")
    lora_dropout: float = Field(0.05, description="LoRA dropout")

    sequence_len: int = Field(1024, description="Maximum sequence length")
    device_map: str = Field("auto", description="Device map for model loading")
    flash_attention: bool = Field(False, description="Use flash attention if available")

    seed: int = Field(42, description="Random seed")
    checkpointing: bool = Field(False, description="Whether to use checkpointing")
    push_to_hub: bool = Field(True, description="Push to HF Hub after training")
    do_validation: bool = Field(False, description="Whether to run validation")
    do_merge: bool = Field(True, description="Whether to merge and push LoRA adapters")
    adapter_subfolder: str = Field("", description="Adapter subfolder in the model repo")

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
