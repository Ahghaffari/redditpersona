"""
runner.py — Orchestrate training across models × strategies.

For each model:
  * one LoRA adapter per grouping strategy (community pooled, label inside the
    system prompt — rendered through the tokenizer's chat template at runtime)
  * one baseline_all adapter trained on all replies regardless of grouping
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List

from config import AppConfig, TrainingCfg
from .dataset import DatasetBuilder, load_assignments
from .trainer import LoRATrainer

logger = logging.getLogger(__name__)


class TrainingRunner:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.tcfg: TrainingCfg = cfg.training
        self.base_dir = Path(cfg.data_dir)
        self.results: List[dict] = []

    def run(self) -> List[dict]:
        device = self.tcfg.device
        if device.startswith("cuda:"):
            os.environ["CUDA_VISIBLE_DEVICES"] = device.split(":")[1]

        builder = DatasetBuilder(self.cfg)

        for ms in self.tcfg.models:
            logger.info("=" * 60)
            logger.info("MODEL: %s", ms.name)
            logger.info("=" * 60)
            t = LoRATrainer(ms, self.tcfg)
            t.load_model()
            try:
                for strategy in self.tcfg.strategies:
                    try:
                        assignments = load_assignments(strategy, self.base_dir)
                    except FileNotFoundError as e:
                        logger.warning("missing assignments: %s — skipping", e)
                        self.results.append({
                            "model": ms.short_name, "strategy": strategy,
                            "community": "pooled", "status": "skipped",
                            "reason": "no_assignments",
                        })
                        continue
                    t0 = time.time()
                    splits = builder.build_strategy_dataset(
                        strategy, assignments, self.tcfg.max_communities,
                    )
                    if splits is None:
                        self.results.append({
                            "model": ms.short_name, "strategy": strategy,
                            "community": "pooled", "status": "skipped",
                            "reason": "too_few_samples",
                        })
                        continue
                    out_dir = t.train(splits, strategy, "pooled")
                    self.results.append({
                        "model": ms.short_name, "strategy": strategy,
                        "community": "pooled", "status": "ok",
                        "train_samples": len(splits["train"]),
                        "adapter_dir": str(out_dir),
                        "elapsed_s": round(time.time() - t0, 1),
                    })

                # baseline_all
                t0 = time.time()
                bs = builder.build_baseline_dataset()
                if bs is not None:
                    out_dir = t.train(bs, "baseline_all", "all")
                    self.results.append({
                        "model": ms.short_name, "strategy": "baseline_all",
                        "community": "all", "status": "ok",
                        "train_samples": len(bs["train"]),
                        "adapter_dir": str(out_dir),
                        "elapsed_s": round(time.time() - t0, 1),
                    })
                else:
                    self.results.append({
                        "model": ms.short_name, "strategy": "baseline_all",
                        "community": "all", "status": "skipped",
                        "reason": "too_few_samples",
                    })
            finally:
                t.unload_model()

        out = Path(self.tcfg.output_dir) / "training_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(self.results, f, indent=2)
        logger.info("Training manifest → %s", out)
        return self.results


def run_training(cfg: AppConfig) -> List[dict]:
    return TrainingRunner(cfg).run()
