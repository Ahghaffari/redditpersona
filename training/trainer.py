"""
trainer.py — QLoRA SFT with PEFT + trl SFTTrainer.

One LoRATrainer instance handles a single base model loaded once; call
.train(splits, strategy, community) repeatedly to fine-tune on different
datasets while keeping the base 4-bit weights resident in GPU memory.
"""

from __future__ import annotations

import gc
import json
import logging
import os
from pathlib import Path
from typing import Dict

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

from config import ModelSpec, TrainingCfg

logger = logging.getLogger(__name__)


class LoRATrainer:
    def __init__(self, model_spec: ModelSpec, tcfg: TrainingCfg):
        self.model_spec = model_spec
        self.tcfg = tcfg
        self._model = None
        self._tokenizer = None

    # model lifecycle
    def _bnb(self) -> BitsAndBytesConfig:
        dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}.get(
            self.tcfg.bnb_4bit_compute_dtype, torch.float16,
        )
        return BitsAndBytesConfig(
            load_in_4bit=self.tcfg.load_in_4bit,
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_quant_type=self.tcfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )

    def _lora(self) -> LoraConfig:
        return LoraConfig(
            r=self.tcfg.lora_r,
            lora_alpha=self.tcfg.lora_alpha,
            lora_dropout=self.tcfg.lora_dropout,
            target_modules=self.tcfg.lora_target_modules,
            bias="none",
            task_type="CAUSAL_LM",
        )

    def load_model(self) -> None:
        if self._model is not None:
            return
        logger.info("Loading model: %s", self.model_spec.name)
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_spec.name,
            cache_dir=self.tcfg.model_cache_dir,
            trust_remote_code=True,
            token=self.tcfg.hf_token,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_spec.name,
            quantization_config=self._bnb(),
            device_map={"": 0},
            cache_dir=self.tcfg.model_cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            token=self.tcfg.hf_token,
        )
        self._model = prepare_model_for_kbit_training(self._model)
        self._model = get_peft_model(self._model, self._lora())

        # Force fp16 — our default GPU may not support bf16.
        for p in self._model.parameters():
            if p.dtype == torch.bfloat16:
                p.data = p.data.to(torch.float16)
        for b in self._model.buffers():
            if b.dtype == torch.bfloat16:
                b.data = b.data.to(torch.float16)



        trainable, total = self._model.get_nb_trainable_parameters()
        logger.info(
            "Loaded %s | trainable %s / %s (%.2f%%)",
            self.model_spec.short_name,
            f"{trainable:,}", f"{total:,}",
            100 * trainable / total,
        )

    def unload_model(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # train one (strategy, community)
    def train(
        self, splits: Dict[str, Dataset], strategy: str, community: str,
    ) -> Path:
        self.load_model()
        adapter_dir = (
            Path(self.tcfg.output_dir)
            / self.model_spec.short_name / strategy / f"community_{community}"
        )
        os.makedirs(adapter_dir, exist_ok=True)

        args = SFTConfig(
            output_dir=str(adapter_dir),
            num_train_epochs=self.tcfg.num_epochs,
            per_device_train_batch_size=self.tcfg.per_device_train_batch_size,
            gradient_accumulation_steps=self.tcfg.gradient_accumulation_steps,
            learning_rate=self.tcfg.learning_rate,
            warmup_ratio=self.tcfg.warmup_ratio,
            weight_decay=self.tcfg.weight_decay,
            lr_scheduler_type=self.tcfg.lr_scheduler_type,
            logging_steps=self.tcfg.logging_steps,
            save_strategy=self.tcfg.save_strategy,
            fp16=self.tcfg.fp16,
            bf16=self.tcfg.bf16,
            optim="paged_adamw_8bit",
            max_grad_norm=0.3,
            report_to="none",
            eval_strategy="no",
            load_best_model_at_end=False,
            remove_unused_columns=True,
            dataloader_pin_memory=False,
            max_length=self.tcfg.max_seq_length,
            gradient_checkpointing=True,
        )
        trainer = SFTTrainer(
            model=self._model,
            processing_class=self._tokenizer,
            train_dataset=splits["train"],
            args=args,
        )
        logger.info(
            "Training: model=%s strategy=%s community=%s n_train=%d",
            self.model_spec.short_name, strategy, community, len(splits["train"]),
        )
        trainer.train()

        self._model.save_pretrained(str(adapter_dir))
        self._tokenizer.save_pretrained(str(adapter_dir))
        with open(adapter_dir / "training_meta.json", "w") as f:
            json.dump({
                "model": self.model_spec.name,
                "strategy": strategy,
                "community": community,
                "train_samples": len(splits["train"]),
                "val_samples":   len(splits.get("val", [])),
                "test_samples":  len(splits.get("test", [])),
            }, f, indent=2)
        logger.info("Adapter saved → %s", adapter_dir)
        return adapter_dir
