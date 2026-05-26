"""
generator.py — Reply generation with QLoRA + optional LoRA adapter.

Conversation rendering is delegated to the tokenizer's chat template via
`apply_chat_template`, so the same code works for any instruct model
(Qwen / Llama-3 / Mistral / Gemma / …).
"""

from __future__ import annotations

import gc
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import ModelSpec, TrainingCfg

logger = logging.getLogger(__name__)


Message = Dict[str, str]


class ReplyGenerator:
    def __init__(self, model_spec: ModelSpec, tcfg: TrainingCfg):
        self.model_spec = model_spec
        self.tcfg = tcfg
        self._model = None
        self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    # tokenizer-only

    def load_tokenizer_only(self) -> None:
        if self._tokenizer is not None:
            return
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_spec.name,
            cache_dir=self.tcfg.model_cache_dir,
            trust_remote_code=True,
            token=self.tcfg.hf_token,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
            self._tokenizer.pad_token_id = self._tokenizer.eos_token_id
        self._tokenizer.padding_side = "left"

    def load(self, adapter_path: Optional[str] = None) -> None:
        self.unload()
        self.load_tokenizer_only()

        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
        base = AutoModelForCausalLM.from_pretrained(
            self.model_spec.name,
            quantization_config=bnb,
            device_map={"": 0},
            cache_dir=self.tcfg.model_cache_dir,
            trust_remote_code=True,
            torch_dtype=torch.float16,
            token=self.tcfg.hf_token,
        )
        if adapter_path:
            self._model = PeftModel.from_pretrained(base, adapter_path)
            logger.info("Loaded adapter: %s", adapter_path)
        else:
            self._model = base
            logger.info("Loaded base model: %s", self.model_spec.name)
        self._model.eval()
        for p in self._model.parameters():
            if p.dtype == torch.bfloat16:
                p.data = p.data.to(torch.float16)
        for b in self._model.buffers():
            if b.dtype == torch.bfloat16:
                b.data = b.data.to(torch.float16)

    def prepare_eval_data(
        self, messages_list: List[List[Message]],
    ) -> Tuple[List[str], List[str], List[List[Message]]]:
        """Convert message lists into (prompts, references, full_messages).

        - prompts: rendered with `add_generation_prompt=True` (assistant turn dropped)
        - references: gold assistant content
        - full_messages: original list, kept for perplexity computation
        """
        self.load_tokenizer_only()
        prompts: List[str] = []
        refs: List[str] = []
        fulls: List[List[Message]] = []
        for msgs in messages_list:
            if not msgs or msgs[-1].get("role") != "assistant":
                continue
            ref = (msgs[-1].get("content") or "").strip()
            if not ref:
                continue
            prompt = self._tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True,
            )
            prompts.append(prompt)
            refs.append(ref)
            fulls.append(msgs)
        return prompts, refs, fulls

    def generate_replies(
        self, prompts: List[str], max_new_tokens: int = 256,
        temperature: float = 0.7, top_p: float = 0.9, do_sample: bool = True,
        batch_size: int = 4,
    ) -> List[str]:
        replies: List[str] = []
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            inputs = self._tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=self.tcfg.max_seq_length,
            ).to(self._model.device)
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    temperature=temperature if do_sample else 1.0,
                    top_p=top_p if do_sample else 1.0, do_sample=do_sample,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id,
                )
            for j, output in enumerate(outputs):
                prompt_len = inputs["input_ids"][j].shape[0]
                gen_ids = output[prompt_len:]
                replies.append(
                    self._tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                )
        return replies

    def compute_perplexity(
        self, messages_list: List[List[Message]], max_length: int = 1024,
    ) -> float:
        """Token-level perplexity over the *assistant* span only.

        Uses the tokenizer's chat template to render both the full conversation
        and the prompt-only prefix; tokens belonging to the prefix are masked
        out (-100) so loss is computed solely on the assistant reply.
        """
        total_loss, total_tokens = 0.0, 0
        for msgs in messages_list:
            if not msgs or msgs[-1].get("role") != "assistant":
                continue
            full_str = self._tokenizer.apply_chat_template(
                msgs, tokenize=False,
            )
            prompt_str = self._tokenizer.apply_chat_template(
                msgs[:-1], tokenize=False, add_generation_prompt=True,
            )
            full_enc = self._tokenizer(
                full_str, truncation=True, max_length=max_length, return_tensors="pt",
            )
            prompt_enc = self._tokenizer(
                prompt_str, truncation=True, max_length=max_length, return_tensors="pt",
            )
            input_ids = full_enc["input_ids"].to(self._model.device)
            labels = input_ids.clone()
            labels[0, : prompt_enc["input_ids"].shape[1]] = -100
            n_resp = (labels[0] != -100).sum().item()
            if n_resp == 0:
                continue
            with torch.no_grad():
                out = self._model(input_ids=input_ids, labels=labels)
            total_loss += out.loss.item() * n_resp
            total_tokens += n_resp
        if total_tokens == 0:
            return float("inf")
        return float(np.exp(total_loss / total_tokens))

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
