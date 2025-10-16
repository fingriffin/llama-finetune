import yaml
from pydantic import BaseModel, Field, field_validator
from pathlib import Path

class FinetuneConfig(BaseModel):
    """Configuration for LoRA finetuning"""

    model_name: str = Field(..., description="Name of the model to use")
    adapter: str = Field(..., description="Name of the adpter model to use")

    load_in_8bit: bool = Field(False, description="Load the model from 8bit")
    load_in_4bit: bool = Field(False, description="Load the model from 4bit")
    bf16: bool = Field(False, description="Load the model from BF16")
    fp16: bool = Field(True, description="Load the model from FP16")
    gradient_checkpointing: bool = Field(True, description="Use gradient checkpointing to save memory")

    optimizer: str = Field("paged_adamw_32bit", description="Optimizer to use")
    gpus: int = Field(1, description="Number of GPUs to use")

    train_data_path: str = Field(..., description="Path to training data")
    val_data_path: str | None = Field(None, description="Path to validation data (optional)")

    output_dir: str = Field(..., description="Directory to save checkpoints and outputs")
    epochs: int = Field(3, description="Number of training epochs")

    micro_batch_size: int = Field(2, description="Batch size per device")
    gradient_accumulation_steps: int = Field(4, description="Number of gradient accumulation steps")
    learning_rate: float = Field(2e-4, description="Learning rate")

    lora_r: int = Field(8, description="LoRA rank")
    lora_alpha: int = Field(16, description="LoRA alpha")
    lora_dropout: float = Field(0.05, description="LoRA dropout")

    sequence_len: int = Field(1024, description="Maximum sequence length")
    device_map: str = Field("auto", description="Device map for model loading")
    flash_attention: bool = Field(False, description="Use flash attention if available")

    seed: int = Field(42, description="Random seed")
    push_to_hub: bool = Field(False, description="Whether to push the adaptor to Hugging Face Hub after training")
    do_validation: bool = Field(False, description="Whether to run validation")

    @field_validator("output_dir")
    def create_output_path(cls, v):
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

def load_finetune_config(config_path: str) -> FinetuneConfig:
    """Load and validate finetuning configuration from YAML file."""
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)
    return FinetuneConfig(**config_dict)