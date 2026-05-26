"""
Strategy 1: Subreddit-level grouping (baseline).

Each community equals one subreddit; every user is assigned to all subs
they participated in (multi-membership).
"""

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))  # add project root to path

import config
from utils import ensure_dirs, write_json, read_json

ensure_dirs()
logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
)
log = logging.getLogger(__name__)

STRATEGY_ID   = "strategy_1_subreddit"
OUTPUT_DIR    = config.DATA_DIR / "groupings" / STRATEGY_ID


def run(user_index: list[dict]) -> dict:
    """
    Assign each user to communities (= subreddits they were active in).

    Parameters
    ----------
    user_index : list[dict]
        Output of Step 2. Each entry has at minimum:
          "username", "subreddits" (list of subreddit names)

    Returns
    -------
    dict with keys:
      "assignments"  — user -> list of community_ids (subreddit names)
      "communities"  — community_id -> {members, category, subreddit_name}
      "meta"         — strategy parameters
    """
    # user -> [subreddit, ...]
    assignments: dict[str, list[str]] = {}
    # community_id -> members
    community_members: dict[str, list[str]] = defaultdict(list)

    for user_profile in user_index:
        username  = user_profile["username"]
        sub_list  = user_profile.get("subreddits", [])
        # Order by activity (posts+comments) desc so assignment[0] is the
        # user's most-active subreddit, not the alphabetically-first one.
        ap = config.PROFILES_DIR / username / "activity.json"
        if ap.exists():
            try:
                act = read_json(ap)
                def _w(s: str) -> int:
                    r = act.get(s, {})
                    return int(r.get("posts", 0)) + int(r.get("comments", 0))
                sub_list = sorted(sub_list, key=lambda s: (-_w(s), s))
            except Exception:
                pass
        assignments[username] = sub_list
        for sub in sub_list:
            community_members[sub].append(username)

    # Build community profiles
    community_profiles = []
    for community_id, members in sorted(community_members.items()):
        community_profiles.append({
            "community_id":   community_id,
            "subreddit_name": community_id,
            "category":       config.SUB_TO_CATEGORY.get(community_id, "unknown"),
            "n_members":      len(members),
            "members":        members,
        })

    community_profiles.sort(key=lambda c: c["n_members"], reverse=True)

    return {
        "assignments": assignments,
        "communities": community_profiles,
        "meta": {
            "strategy":      STRATEGY_ID,
            "n_users":       len(assignments),
            "n_communities": len(community_profiles),
            "description":   "Each subreddit = one community (baseline)",
            "run_at":        datetime.now(timezone.utc).isoformat(),
        },
    }


def main():
    user_index_path = config.DATA_DIR / "user_index.json"
    if not user_index_path.exists():
        log.error("user_index.json not found. Run `python run.py profiles` first.")
        return

    user_index = read_json(user_index_path)
    log.info(f"[Strategy 1] Running on {len(user_index)} users")

    result = run(user_index)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "community_assignments.json", result["assignments"])
    write_json(OUTPUT_DIR / "community_profiles.json",   result["communities"])
    write_json(OUTPUT_DIR / "grouping_meta.json",        result["meta"])

    log.info(
        f"[Strategy 1] Done — {result['meta']['n_communities']} communities, "
        f"{result['meta']['n_users']} users"
    )
    log.info(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
