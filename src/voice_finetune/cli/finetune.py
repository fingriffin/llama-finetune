"""Finetune a model based on a given config file."""

import os
from datetime import datetime
from pathlib import Path

import click
import wandb
from axolotl.cli.config import load_cfg
from axolotl.cli.preprocess import PreprocessCliArgs, do_preprocess
from axolotl.common.datasets import load_datasets
from axolotl.train import train
from axolotl.utils.dict import DictDefault
from dotenv import load_dotenv
from loguru import logger

from voice_finetune.config import load_finetune_config
from voice_finetune.hf import configure_hf, get_token
from voice_finetune.logging import setup_logging


@click.command()
@click.argument("config_path")
@click.option("--log-level", default="INFO", help="Logging level")
@click.option("--log-file", help="Log file path")
@click.option("--model-name", help="Override model name")
@click.option("--train-data-path", help="Override training data path")
@click.option("--output-dir", help="Override output directory")
@click.option("--epochs", type=int, help="Override number of epochs")
@click.option("--micro-batch-size", type=int, help="Override micro batch size")
@click.option("--learning-rate", type=float, help="Override learning rate")
@click.option("--optimizer", help="Override optimizer")
@click.option("--seed", type=int, help="Override random seed")
@click.option("--device-map", help="Override device map (e.g., 'auto', 'cpu', 'cuda')")
@click.option("--bf16/--no-bf16", default=None, help="Override bf16 usage")
@click.option("--fp16/--no-fp16", default=None, help="Override fp16 usage")
@click.option("--load-in-8bit/--no-load-in-8bit", default=None, help="Override 8b quant")
def main(
    config_path: str,
    log_level: str,
    log_file: str | None,
    model_name: str,
    train_data_path: str,
    output_dir: str,
    epochs: int,
    micro_batch_size: int,
    learning_rate: float,
    optimizer: str,
    seed: int,
    device_map: str,
    bf16: bool,
    fp16: bool,
    load_in_8bit: bool,
) -> None:
    """
    Run finetuning job based on the provided config file.

    :param config_path: Path to the config file.
    :param log_level: Optional override for logging level.
    :param log_file: Optional override for log file path.
    :param model_name: Optional override for model name.
    :param train_data_path: Optional override for training data path.
    :param output_dir: Optional override for output directory.
    :param epochs: Optional override for number of epochs.
    :param micro_batch_size: Optional override for micro batch size.
    :param learning_rate: Optional override for learning rate.
    :param optimizer: Optional override for optimizer.
    :param seed: Optional override for random seed.
    :param device_map: Optional override for device map.
    :param bf16: Optional override for bf16 usage.
    :param fp16: Optional override for fp16 usage.
    :param load_in_8bit: Optional override for 8-bit loading.
    :return: None

    :raises Exception: If loading the finetune config fails.
    """
    # Setup logging
    setup_logging(level=log_level, log_file=log_file)

    # Load config
    try:
        logger.info("Loading config from {}", config_path)
        config = load_finetune_config(config_path)
        logger.success("Config loaded successfully!")
        print("Current configuration:")
        print(config.model_dump_json(indent=2))
        print("")
    except Exception as e:
        logger.error("Failed to load config: {}", e)
        raise

    # Apply overrides if provided
    if model_name:
        config.model_name = model_name
    if train_data_path:
        config.train_data_path = train_data_path
    if output_dir:
        config.output_dir = output_dir
    if epochs is not None:
        config.epochs = epochs
    if micro_batch_size is not None:
        config.micro_batch_size = micro_batch_size
    if learning_rate is not None:
        config.learning_rate = learning_rate
    if optimizer:
        config.optimizer = optimizer
    if seed is not None:
        config.seed = seed
    if device_map:
        config.device_map = device_map
    if bf16 is not None:
        config.bf16 = bf16
    if fp16 is not None:
        config.fp16 = fp16
    if load_in_8bit is not None:
        config.load_in_8bit = load_in_8bit

    # Configure HF environment
    configure_hf(config.model_name)
    get_token()
    if config.push_to_hub:
        model_name = os.path.basename(config.output_dir.rstrip("/"))
        hub_model_id = f"{os.getenv('HF_ORG')}/{model_name}"
        if config.checkpointing:
            hub_strategy = "every_save"
        else:
            hub_strategy = "end"
        logger.info("Will push adapter to the Hub with model ID: {}", hub_model_id)
    else:
        hub_model_id = None
        hub_strategy = None

    # Configure W&B
    load_dotenv()
    wandb.login(key=os.getenv("WANDB_API_KEY"))

    # Resolve data path
    hf_org = os.getenv("HF_ORG")
    if hf_org:
        if str(config.train_data_path).startswith(hf_org + '/'):
            data_path = config.train_data_path
            logger.info(f"Detected Hugging Face dataset: {str(config.train_data_path)}")
    else:
        data_path = Path(config.train_data_path).expanduser().resolve() # type: ignore[assignment]
        if not data_path.exists(): # type: ignore[attr-defined]
            logger.error("Training data not found at: {}", str(data_path))
            raise

    # Configure checkpointing strategy
    if config.checkpointing:
        logger.info("Checkpointing enabled: will save model at the end of each epoch.")
        save_strategy = "epoch"
        save_total_limit = config.epochs
        save_only_model = False
    else:
        logger.info("Checkpointing disabled: will only save final model.")
        save_steps = 0
        save_strategy = "no"
        save_total_limit = 0
        save_only_model = True

    logger.info("Starting finetuning job...")

    # Get current timestamp for W&B run name
    now = datetime.now().strftime('%Y%m%d_%H%M')
    experiment_base_name = os.path.basename(config_path.replace('configs/', ''))

    # Convert config to axolotl config
    axolotl_cfg_raw = DictDefault(
        base_model=config.model_name,
        seed=config.seed,
        output_dir=config.output_dir,
        device_map=config.device_map,

        adapter=config.adapter,
        load_in_8bit=config.load_in_8bit,
        load_in_4bit=config.load_in_4bit,
        bf16=config.bf16,
        fp16=config.fp16,
        optimizer=config.optimizer,
        num_epochs=config.epochs,
        learning_rate=config.learning_rate,
        micro_batch_size=config.micro_batch_size,
        sequence_len=config.sequence_len,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        flash_attention=config.flash_attention,

        lora_r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        lora_target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],

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

        test_datasets=[
            *(
                [{
                    "path": str(data_path),
                    "split": "validation",
                    "type": "chat_template",
                    "field_messages": "messages",
                    "message_field_role": "from",
                    "message_field_content": "value",
                }]
                if config.do_validation else []
            )
        ],
        eval_steps = 1,

        use_wandb=True,
        wandb_project=os.getenv('WANDB_PROJECT'),
        wandb_entity=os.getenv('WANDB_ENTITY'),
        wandb_name=f"{os.path.splitext(experiment_base_name)[0]}_{now}",
        wandb_watch="checkpoint",
        wandb_log_model="checkpoint",
        hub_model_id=hub_model_id,
        hub_strategy=hub_strategy,
    )

    # Add pad token only if model is a Llama
    if "llama" in config.model_name.lower():
        axolotl_cfg_raw.setdefault("special_tokens", {})["pad_token"] = "<PAD>"

    axolotl_cfg = load_cfg(axolotl_cfg_raw)

    # Preprocess config for VRAM stability
    logger.info("Preprocessing axolotl config...")
    cli_args = PreprocessCliArgs()
    do_preprocess(axolotl_cfg, cli_args)

    # Load dataset
    train_dataset = load_datasets(cfg=axolotl_cfg)
    logger.info("Training dataset loaded from {}", str(data_path))

    # Training
    model, tokenizer, trainer = train(cfg=axolotl_cfg, dataset_meta=train_dataset)
