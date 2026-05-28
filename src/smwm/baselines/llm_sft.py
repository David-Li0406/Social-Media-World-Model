"""SFT (Supervised Fine-Tuning) baseline for Qwen3-4B via OpenRLHF.

Project CLAUDE.md mandates OpenRLHF for LLM training; meta-skill CLAUDE.md
mandates GPUs 0,1 and storing large artifacts (LoRA adapters, checkpoints)
under /scratch/daweili5/. This baseline honours both:

  - fit():     spawns `deepspeed ... openrlhf.cli.train_sft ...` on GPUs 0,1,
               writes the LoRA adapter to `adapter_path` (default under
               /scratch/daweili5/smwm/runs/<run_name>/adapter).
  - predict(): loads the base model + LoRA via peft and reuses the same JSON
               prompt-and-parse loop as the zero-shot `llm` baseline.

Heavy deps (torch, transformers, peft, deepspeed) are imported lazily so the
module is importable on CPU-only / minimal environments. fit() runs the
training in a subprocess so Python state stays clean (DeepSpeed pollutes
process state aggressively).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from json.decoder import JSONDecodeError
from pathlib import Path
from typing import Any

from .base import Baseline
from .registry import register


DEFAULT_SCRATCH = "/scratch/daweili5/smwm"


@register("llm_sft")
class LLMSFTBaseline(Baseline):
    def __init__(
        self,
        model_id: str = "Qwen/Qwen3-4B",
        run_name: str = "sft_qwen3_4b",
        adapter_path: str | None = None,
        # data
        max_len: int = 4096,
        # optimization
        epochs: int = 2,
        learning_rate: float = 2.0e-4,
        train_batch_size: int = 16,
        micro_train_batch_size: int = 1,
        # LoRA
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        target_modules: str = "q_proj k_proj v_proj o_proj gate_proj up_proj down_proj",
        # backend
        gpus: str = "0,1",
        bf16: bool = True,
        gradient_checkpointing: bool = True,
        flash_attn: bool = False,
        # inference
        max_new_tokens: int = 500,
        retries: int = 3,
        enable_thinking: bool = False,
        load_in_4bit: bool = False,
        # toggles
        skip_train_if_adapter_exists: bool = True,
        extra_args: list[str] | None = None,
        **kwargs,
    ):
        self.model_id = model_id
        self.run_name = run_name
        self.adapter_path = Path(
            adapter_path
            if adapter_path
            else f"{DEFAULT_SCRATCH}/runs/{run_name}/adapter"
        )
        self.max_len = int(max_len)
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.train_batch_size = int(train_batch_size)
        self.micro_train_batch_size = int(micro_train_batch_size)
        self.lora_rank = int(lora_rank)
        self.lora_alpha = int(lora_alpha)
        self.lora_dropout = float(lora_dropout)
        self.target_modules = target_modules
        self.gpus = gpus
        self.bf16 = bool(bf16)
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.flash_attn = bool(flash_attn)
        self.max_new_tokens = int(max_new_tokens)
        self.retries = int(retries)
        self.enable_thinking = bool(enable_thinking)
        self.load_in_4bit = bool(load_in_4bit)
        self.skip_train_if_adapter_exists = bool(skip_train_if_adapter_exists)
        self.extra_args = list(extra_args) if extra_args else []
        self._tokenizer = None
        self._model: Any = None

    # ------------------------------------------------------------------ fit
    def _write_openrlhf_dataset(self, records: list[dict], path: Path) -> int:
        """OpenRLHF SFT expects JSONL with input/output text fields."""
        path.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with open(path, "w", encoding="utf-8") as f:
            for r in records:
                prompt = r.get("prompt")
                completion = r.get("completion")
                if prompt is None or completion is None:
                    continue
                f.write(
                    json.dumps(
                        {"prompt": prompt, "response": completion},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                n += 1
        return n

    def _build_train_command(self, train_jsonl: Path, eval_jsonl: Path | None) -> list[str]:
        cmd = [
            "deepspeed",
            f"--include=localhost:{self.gpus}",
            "--module",
            "openrlhf.cli.train_sft",
            "--save_path",
            str(self.adapter_path),
            "--ckpt_path",
            str(self.adapter_path / "ckpt"),
            "--dataset",
            str(train_jsonl),
            "--input_key",
            "prompt",
            "--output_key",
            "response",
            "--pretrain",
            self.model_id,
            "--max_len",
            str(self.max_len),
            "--max_epochs",
            str(self.epochs),
            "--learning_rate",
            str(self.learning_rate),
            "--train_batch_size",
            str(self.train_batch_size),
            "--micro_train_batch_size",
            str(self.micro_train_batch_size),
            "--lora_rank",
            str(self.lora_rank),
            "--lora_alpha",
            str(self.lora_alpha),
            "--lora_dropout",
            str(self.lora_dropout),
            "--target_modules",
            *self.target_modules.split(),
            "--zero_stage",
            "2",
            "--save_steps",
            "-1",
            "--logging_steps",
            "10",
            "--eval_steps",
            "-1",
        ]
        if self.bf16:
            cmd.append("--bf16")
        if self.gradient_checkpointing:
            cmd.append("--gradient_checkpointing")
        if self.flash_attn:
            cmd.append("--flash_attn")
        if eval_jsonl is not None:
            cmd += ["--eval_dataset", str(eval_jsonl)]
        cmd += self.extra_args
        return cmd

    def fit(self, train_records: list[dict]) -> None:
        if self.skip_train_if_adapter_exists and (self.adapter_path / "adapter_model.safetensors").exists():
            print(f"[llm_sft] adapter already present at {self.adapter_path}; skipping training")
            return

        self.adapter_path.mkdir(parents=True, exist_ok=True)

        # Materialize the OpenRLHF dataset. We always re-write so the file
        # reflects the current train_records argument (smoke vs. full).
        train_jsonl = self.adapter_path / "train.jsonl"
        n_train = self._write_openrlhf_dataset(train_records, train_jsonl)
        if n_train == 0:
            raise RuntimeError("no usable training records (missing prompt/completion)")

        cmd = self._build_train_command(train_jsonl=train_jsonl, eval_jsonl=None)
        log_path = self.adapter_path / "train.log"
        print(f"[llm_sft] launching: {' '.join(cmd)}")
        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", self.gpus)
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        with open(log_path, "w") as logf:
            proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
        if proc.returncode != 0:
            tail = log_path.read_text().splitlines()[-40:]
            raise RuntimeError(
                f"OpenRLHF SFT failed (rc={proc.returncode}). Last 40 lines of "
                f"{log_path}:\n" + "\n".join(tail)
            )
        print(f"[llm_sft] training done; adapter at {self.adapter_path}")

    # ---------------------------------------------------------------- predict
    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from huggingface_hub import login
        from peft import PeftModel
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        token = os.getenv("HUGGING_FACE")
        if token:
            login(token)
        assert torch.cuda.is_available(), "No CUDA device found"

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        load_kwargs: dict = {"device_map": "cuda:0"}
        if self.load_in_4bit:
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
        else:
            load_kwargs["torch_dtype"] = torch.bfloat16

        base = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        if (self.adapter_path / "adapter_config.json").exists():
            self._model = PeftModel.from_pretrained(base, str(self.adapter_path))
            print(f"[llm_sft] loaded LoRA adapter from {self.adapter_path}")
        else:
            print(
                f"[llm_sft] WARNING: no adapter at {self.adapter_path}; "
                "predicting with the un-tuned base model."
            )
            self._model = base
        self._model.eval()

    def _generate_json(self, prompt: str) -> tuple[dict | None, str]:
        import torch

        self._load()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        messages = [{"role": "user", "content": prompt}]
        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
        )
        inputs = self._tokenizer([text], return_tensors="pt").to(self._model.device)
        raw = ""
        for _ in range(self.retries):
            gen = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            out_ids = gen[0][len(inputs.input_ids[0]):].tolist()
            raw = self._tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            try:
                return json.loads(raw), raw
            except JSONDecodeError:
                continue
        return None, raw

    def predict(self, record: dict) -> dict:
        prompt = record.get("prompt")
        if prompt is None:
            from ..data.build_splits import make_prompt
            from .base import get_context, get_stimulus

            prompt = make_prompt(get_context(record), get_stimulus(record))
        parsed, raw = self._generate_json(prompt)
        if parsed is None:
            return {
                "score": 0,
                "width": 0,
                "controversiality": 0,
                "reply_summary": "",
                "_raw": raw,
            }
        return {
            "score": int(parsed.get("score", 0)),
            "width": int(parsed.get("width", 0)),
            "controversiality": int(parsed.get("controversiality", 0)),
            "reply_summary": str(parsed.get("reply_summary", "")),
            "_raw": raw,
        }
