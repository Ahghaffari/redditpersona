"""
Single-strategy training test.

Trains exactly one (model, strategy) pair to validate the full pipeline:
dataset build → tokenizer chat template → SFTTrainer → adapter saved.

Defaults: strategy_3_semantic (only 5 communities, smallest pooled set),
1 epoch, capped sample count, fast logging cadence.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("training test")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--strategy", default="strategy_3_semantic")
    ap.add_argument("--max-samples", type=int, default=512,
                    help="Cap pooled samples before train/val/test split")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--device", default="cuda:1")
    args = ap.parse_args()

    from config import load_config
    cfg = load_config(args.config)

    # Override for fast training test
    cfg.training.num_epochs = args.epochs
    cfg.training.device = args.device
    cfg.training.logging_steps = 5
    cfg.training.save_strategy = "no"
    cfg.training.output_dir = str(Path(cfg.training.output_dir) / "_training")

    if args.device.startswith("cuda:"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.split(":")[1]

    from training.dataset import DatasetBuilder, load_assignments
    from training.trainer import LoRATrainer

    base_dir = Path(cfg.data_dir)
    log.info("Loading assignments for %s ...", args.strategy)
    assignments = load_assignments(args.strategy, base_dir)

    builder = DatasetBuilder(cfg)
    log.info("Building pooled dataset (max_communities=%d) ...",
             cfg.training.max_communities)
    splits = builder.build_strategy_dataset(
        args.strategy, assignments, cfg.training.max_communities,
    )
    if splits is None:
        log.error("Dataset build returned None — abort.")
        sys.exit(1)

    # Cap samples for training test
    n_train = min(args.max_samples, len(splits["train"]))
    n_val = min(max(8, args.max_samples // 8), len(splits["val"]))
    splits = {
        "train": splits["train"].select(range(n_train)),
        "val":   splits["val"].select(range(n_val)),
        "test":  splits["test"],
    }
    log.info("Capped: train=%d val=%d test=%d", n_train, n_val, len(splits["test"]))

    # Inspect one sample to confirm chat-template-ready format
    sample = splits["train"][0]
    log.info("Sample messages structure:")
    for msg in sample["messages"]:
        content = msg["content"]
        log.info("  [%s] %s%s", msg["role"], content[:80],
                 "…" if len(content) > 80 else "")

    ms = cfg.training.models[0]
    log.info("Model: %s", ms.name)

    t0 = time.time()
    trainer = LoRATrainer(ms, cfg.training)
    trainer.load_model()
    try:
        out_dir = trainer.train(splits, args.strategy, "pooled_training_test")
    finally:
        trainer.unload_model()

    elapsed = time.time() - t0
    log.info("=" * 60)
    log.info("training TEST PASSED")
    log.info("  adapter: %s", out_dir)
    log.info("  elapsed: %.1fs", elapsed)
    log.info("=" * 60)

    # List artefacts
    files = sorted(p.name for p in out_dir.iterdir())
    log.info("Adapter files: %s", files)


if __name__ == "__main__":
    main()
