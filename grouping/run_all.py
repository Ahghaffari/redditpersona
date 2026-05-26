"""Run all six grouping strategies in sequence."""

import argparse
import logging
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import read_json, write_json

from grouping.strategy_subreddit   import run as run_subreddit
from grouping.strategy_graph       import main as main_graph
from grouping.strategy_semantic    import main as main_semantic
from grouping.strategy_hybrid      import main as main_hybrid
from grouping.strategy_interaction import main as main_interaction

logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Run grouping strategies")
    parser.add_argument(
        "--strategies", nargs="*", type=int, default=[1, 2, 3, 4, 5],
        help="Which strategies to run (default: all)",
    )
    args = parser.parse_args()

    user_index_path = config.DATA_DIR / "user_index.json"
    if not user_index_path.exists():
        log.error("user_index.json not found. Run `python run.py profiles` first.")
        return
    user_index = read_json(user_index_path)
    log.info(f"Loaded {len(user_index)} users")

    if 1 in args.strategies:
        log.info("=" * 40 + " STRATEGY 1 (subreddit) " + "=" * 40)
        r = run_subreddit(user_index)
        out = config.DATA_DIR / "groupings" / "strategy_1_subreddit"
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "community_assignments.json", r["assignments"])
        write_json(out / "community_profiles.json",   r["communities"])
        write_json(out / "grouping_meta.json",        r["meta"])
        log.info(f"Strategy 1 done — {r['meta']['n_communities']} communities")

    if 2 in args.strategies:
        log.info("=" * 40 + " STRATEGY 2 (graph) " + "=" * 40)
        main_graph()

    if 3 in args.strategies:
        log.info("=" * 40 + " STRATEGY 3 (semantic) " + "=" * 40)
        main_semantic()

    if 4 in args.strategies:
        log.info("=" * 40 + " STRATEGY 4 (hybrid) " + "=" * 40)
        main_hybrid()

    if 5 in args.strategies:
        log.info("=" * 40 + " STRATEGY 5 (interaction) " + "=" * 40)
        main_interaction()

    log.info("All requested strategies complete. Next: python run.py analysis")


if __name__ == "__main__":
    main()
