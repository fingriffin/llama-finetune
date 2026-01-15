"""Finetuning class for finetuning with Axolotl."""

import json
import os
import select
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import torch
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

PACKAGE_DIR = Path(__file__).parent

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

        self.vllm_devices: str = ""
        self.training_devices: str = ""

        self.config: FinetuneConfig | None = None

        self._prepare_config()
        self._prepare_tokenizer()

    def setup_vllm(self) -> None:
        """
        Set up vLLM engine for fine-tuning.

        :return: None
        """
        if not getattr(self, "config", None) or not getattr(self.config, "vllm", None):
            logger.info("vLLM config not present; skipping vLLM setup.")
            return

        num_vllm_gpus = int(self.config.vllm.tensor_parallel_size or 0)  # type: ignore[union-attr]
        num_gpus = torch.cuda.device_count()

        if num_gpus <= 0:
            raise RuntimeError("No CUDA GPUs detected; cannot start vLLM.")

        if num_vllm_gpus <= 0:
            raise ValueError(
                f"Invalid vLLM tensor_parallel_size: {num_vllm_gpus} (must be >= 1)."
            )

        if num_vllm_gpus >= num_gpus:
            raise ValueError(
                f"Invalid vLLM GPU count: {num_vllm_gpus} (total GPUs: {num_gpus}). "
                "Need at least 1 GPU left for training."
            )

        # GPUs reserved for vLLM (highest indices)
        self.vllm_devices = ",".join(
            str(i) for i in range(num_gpus - num_vllm_gpus, num_gpus)
        )
        logger.info(f"Allocating devices {self.vllm_devices} for vLLM")

        # GPUs reserved for training (lowest indices)
        self.training_devices = ",".join(
            str(i) for i in range(0, num_gpus - num_vllm_gpus)
        )
        logger.info(f"Allocating devices {self.training_devices} for training")

        env = os.environ.copy()

        # Isolate the vLLM process to only the reserved GPUs
        env["CUDA_VISIBLE_DEVICES"] = self.vllm_devices

        # NCCL stability flags for TP init on cloud/virtualised setups (Runpod etc)
        env["NCCL_IB_DISABLE"] = "1"  # disable InfiniBand transport
        env["NCCL_P2P_DISABLE"] = "1"  # disable direct GPU P2P (non NVLink topologies)
        env["NCCL_SHM_DISABLE"] = "1"  # disable shared memory transport

        # Optional debug
        # env["NCCL_DEBUG"] = "INFO"
        # env["VLLM_LOGGING_LEVEL"] = "DEBUG"

        # Pull vLLM launch args from config
        model_id = str(self.config.base_model) # type: ignore[union-attr]
        tp_size = int(num_vllm_gpus)
        gpu_mem_util = float(
            getattr(self.config.vllm, "gpu_memory_utilization", 0.90) or 0.90)  # type: ignore[union-attr]
        max_model_len = int(self.config.vllm.max_model_len or 4096)  # type: ignore[union-attr]
        host = "0.0.0.0"
        port = 8000

        cmd = [
            "trl",
            "vllm-serve",
            "--model",
            model_id,
            "--tensor-parallel-size",
            str(tp_size),
            "--port",
            str(port),
            "--host",
            host,
            "--gpu-memory-utilization",
            str(gpu_mem_util),
            "--max-model-len",
            str(max_model_len),
        ]
        logger.info("Starting external vLLM OpenAI-compatible server process")
        logger.info("vLLM cmd: %s", " ".join(cmd))

        # Start vLLM as a background process
        self.vllm_process = subprocess.Popen(
            cmd,
            env=env,
            cwd=PACKAGE_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        ready_url = f"http://127.0.0.1:{port}/v1/models"
        self.vllm_ready_url = ready_url

        deadline_s = 180.0
        start = time.time()
        last_log_line: str | None = None

        while True:
            if self.vllm_process.poll() is not None:
                output = ""
                try:
                    if self.vllm_process.stdout is not None:
                        output = self.vllm_process.stdout.read()
                except Exception:
                    pass
                raise RuntimeError(
                    "vLLM process exited before becoming ready.\n"
                    f"Last output:\n{output[-4000:]}"
                )

            try:
                if self.vllm_process.stdout is not None:
                    rlist, _, _ = select.select([self.vllm_process.stdout], [], [], 0.0)
                    if rlist:
                        line = self.vllm_process.stdout.readline()
                        if line:
                            last_log_line = line.rstrip()
                            logger.debug("[vLLM] %s", last_log_line)
            except Exception:
                pass

            # Probe the HTTP endpoint.
            try:
                with urllib.request.urlopen(ready_url, timeout=2) as resp:
                    body = resp.read().decode("utf-8")
                    payload = json.loads(body)
                    if isinstance(payload, dict) and payload.get("object") == "list":
                        logger.info("vLLM server is ready at %s", ready_url)
                        break
            except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
                pass

            if (time.time() - start) > deadline_s:
                raise TimeoutError(
                    "Timed out waiting for vLLM server readiness.\n"
                    f"Last vLLM line: {last_log_line}"
                )

            time.sleep(0.25)

    def train(self) -> None:
        """
        Start the finetuning process using Axolotl CLI.

        :return: None
        """
        if not self.local_config_path:
            raise ValueError("axolotl_config_path must be set before training.")

        env = os.environ.copy()

        # If vLLM configured/started, isolate training GPUs
        # If vLLM not configured/started, leave CUDA_VISIBLE_DEVICES unchanged
        vllm_enabled = bool(
            getattr(self, "config", None) and getattr(self.config, "vllm", None)
        )
        vllm_started = bool(getattr(self, "vllm_process", None))

        if vllm_enabled and vllm_started:
            # Ensure vLLM is still alive
            if self.vllm_process.poll() is not None:
                raise RuntimeError(
                    "vLLM process is not running (it exited before training started)."
                )

            # Apply training GPU visibility
            env["CUDA_VISIBLE_DEVICES"] = getattr(self, "training_devices", "")

            # TODO: Use tensor parallelism for training, cf. issue #18
            # train_num_procs = len(self.training_devices.split(",")) \
            #     if getattr(self, "training_devices", "")\
            #     else 1
            # extra_args = ["--num_processes", str(train_num_procs)]
            extra_args = [] # type: ignore[var-annotated]
        else:
            extra_args = []

        # Inject wandb variables only if resuming a run
        if self.wandb_run_id:
            env["WANDB_RESUME"] = "must"
            env["WANDB_RUN_ID"] = self.wandb_run_id

            if self.config:
                if self.config.wandb_project:
                    env["WANDB_PROJECT"] = self.config.wandb_project
                if self.config.wandb_entity:
                    env["WANDB_ENTITY"] = self.config.wandb_entity

        subprocess.run(
            [
                "axolotl",
                "train",
                self.local_config_path,
                *extra_args,
            ],
            check=True,
            env=env,
            cwd=PACKAGE_DIR,
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
