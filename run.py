"""
RedditPersona.

Usage examples (from /home/ubuntu/llm_imitation/Redditpersona/):
    python run.py verify
    python run.py collect
    python run.py profiles
    python run.py grouping --strategies 1 2 3 4 5
    python run.py analysis
    python run.py anonymize           # only if cfg.anonymization.enabled
    python run.py training
    python run.py evaluation
    python run.py all                 # profiles → grouping → analysis → training → evaluation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import config
from config import load_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
log = logging.getLogger("run")

def _stage_verify():
    from collection.verify_subreddits import main as fn
    fn()


def _stage_collect():
    from collection.collect_subreddits import main as fn
    fn()


def _stage_profiles():
    from profiles.build_profiles import main as fn
    fn()


def _stage_grouping(strategies: list[int]):
    from grouping.run_all import main as fn
    sys.argv = ["run_all"] + ["--strategies"] + [str(s) for s in strategies]
    fn()


def _stage_analysis():
    from grouping.analyze import main as fn
    sys.argv = ["analyze"]
    fn()


def _stage_anonymize(cfg_path: str):
    cfg = load_config(cfg_path)
    if not cfg.anonymization.enabled:
        log.info("Anonymization disabled in %s — skipping.", cfg_path)
        return
    from anonymization import Anonymizer
    Anonymizer(cfg).run()


def _stage_training(cfg_path: str):
    cfg = load_config(cfg_path)
    from training import run_training
    run_training(cfg)


def _stage_evaluation(cfg_path: str):
    cfg = load_config(cfg_path)
    from evaluation import run_evaluation
    run_evaluation(cfg)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="redditpersona", description=__doc__)
    p.add_argument("--config", default=str(ROOT / "config.yaml"),
                   help="Path to config.yaml (used by anon/training/evaluation)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("verify",   help="Validate that target subreddits are reachable")
    sub.add_parser("collect",  help="Collect raw posts/comments from Reddit")
    sub.add_parser("profiles", help="Build per-user profiles + text corpora (streaming)")

    g = sub.add_parser("grouping", help="Run grouping strategies (1-5)")
    g.add_argument("--strategies", nargs="*", type=int, default=[1, 2, 3, 4, 5])

    sub.add_parser("analysis",   help="Compute grouping-quality metrics across strategies")
    sub.add_parser("anonymize",  help="Run anonymization (requires cfg.anonymization.enabled)")
    sub.add_parser("training",   help="Run training pipeline (QLoRA)")
    sub.add_parser("evaluation", help="Run evaluation pipeline")

    a = sub.add_parser("all", help="Run profiles → grouping → analysis → training → evaluation")
    a.add_argument("--strategies", nargs="*", type=int, default=[1, 2, 3, 4, 5])
    return p


def main():
    args = _build_parser().parse_args()

    if args.cmd == "verify":      _stage_verify();   return
    if args.cmd == "collect":     _stage_collect();  return
    if args.cmd == "profiles":    _stage_profiles(); return
    if args.cmd == "grouping":    _stage_grouping(args.strategies); return
    if args.cmd == "analysis":    _stage_analysis(); return
    if args.cmd == "anonymize":   _stage_anonymize(args.config); return
    if args.cmd == "training":    _stage_training(args.config);   return
    if args.cmd == "evaluation":  _stage_evaluation(args.config); return
    if args.cmd == "all":
        _stage_profiles()
        _stage_grouping(args.strategies)
        _stage_analysis()
        _stage_anonymize(args.config)
        _stage_training(args.config)
        _stage_evaluation(args.config)
        return


if __name__ == "__main__":
    main()
