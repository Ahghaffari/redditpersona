"""
Step 2: Build Per-User Profiles
================================================================
It is designed to handle the full collected dataset without holding all per-user text in RAM.

Streaming strategy
------------------
1. Aggregate user_activity_matrix.jsonl in one streaming pass.
2. Apply MIN_USER_COMMENTS / MIN_USER_SUBREDDITS thresholds.
3. Pre-create one directory per qualifying user.
4. Single pass over every {sub}/{posts,comments}.jsonl file:
      open the qualifying author's text_corpus.txt in append mode,
      write their snippet, close.
   To avoid the open()/close() per-line cost we maintain an LRU pool of
   open file handles (size = config.PROFILE_OPEN_FILE_POOL).
5. After the streaming pass, walk the profile dirs once to collect text
   stats and emit profile.json / activity.json / user_index.json.

Outputs (per user):
    {PROFILES_DIR}/{user}/text_corpus.txt
    {PROFILES_DIR}/{user}/activity.json
    {PROFILES_DIR}/{user}/profile.json
Plus the global:
    {DATA_DIR}/user_index.json
"""

import logging
import os
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import config
from utils import ensure_dirs, write_json, iter_jsonl

ensure_dirs()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

logging.basicConfig(
    level=config.LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(config.LOGS_DIR / "02_profiles.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


# 1. Streaming aggregation of user_activity_matrix.jsonl
def aggregate_user_activity() -> dict[str, dict[str, dict]]:
    """user → subreddit → {posts, comments, first_utc, last_utc}."""
    if not config.USER_MATRIX_FILE.exists():
        log.warning("user_activity_matrix.jsonl not found. Run Step 1 first.")
        return {}

    aggregated: dict[str, dict[str, dict]] = defaultdict(
        lambda: defaultdict(lambda: {
            "posts": 0, "comments": 0,
            "first_utc": float("inf"), "last_utc": float("-inf"),
        })
    )

    n = 0
    for row in iter_jsonl(config.USER_MATRIX_FILE):
        user = row.get("user", "")
        sub  = row.get("subreddit", "")
        if not user or not sub:
            continue
        rec = aggregated[user][sub]
        rec["posts"]    += row.get("posts", 0)
        rec["comments"] += row.get("comments", 0)
        first = row.get("first_seen_utc")
        last  = row.get("last_seen_utc")
        if first is not None:
            rec["first_utc"] = min(rec["first_utc"], first)
        if last is not None:
            rec["last_utc"]  = max(rec["last_utc"],  last)
        n += 1
        if n % 1_000_000 == 0:
            log.info(f"[Aggregate] {n:,} matrix rows processed, {len(aggregated):,} users")

    log.info(f"[Aggregate] {len(aggregated):,} unique users in activity matrix")
    return dict(aggregated)


# 2. Threshold filter

def filter_users(
    activity: dict[str, dict[str, dict]],
) -> dict[str, dict[str, dict]]:
    qualifying = {
        u: subs for u, subs in activity.items()
        if sum(r["comments"] for r in subs.values()) >= config.MIN_USER_COMMENTS
        and len(subs) >= config.MIN_USER_SUBREDDITS
    }
    log.info(
        f"[Filter] {len(qualifying):,} users pass thresholds "
        f"(min_comments={config.MIN_USER_COMMENTS}, "
        f"min_subs={config.MIN_USER_SUBREDDITS}); "
        f"removed {len(activity) - len(qualifying):,}"
    )
    return qualifying


# 3. LRU pool of open append handles
class _HandlePool:
    """LRU pool of open file handles, capped at `max_open` files."""

    def __init__(self, max_open: int):
        self.max_open = max_open
        self._handles: "OrderedDict[Path, object]" = OrderedDict()

    def get(self, path: Path):
        h = self._handles.get(path)
        if h is not None:
            self._handles.move_to_end(path)
            return h
        # Open in append mode (binary write avoids re-encoding overhead).
        h = open(path, "a", encoding="utf-8")
        self._handles[path] = h
        if len(self._handles) > self.max_open:
            old_path, old_h = self._handles.popitem(last=False)
            old_h.close()
        return h

    def close_all(self):
        for h in self._handles.values():
            try:
                h.close()
            except Exception:
                pass
        self._handles.clear()


# 4. Streaming text-corpus build

def stream_text_corpora(qualifying: set[str]):
    """One pass over every {sub}/{posts,comments}.jsonl."""
    if not config.SUBS_DIR.exists():
        log.warning("subreddits/ dir not found. Run Step 1 first.")
        return

    # Pre-create output dirs and truncate any pre-existing corpora.
    for user in qualifying:
        d = config.PROFILES_DIR / user
        d.mkdir(parents=True, exist_ok=True)
        # Truncate so reruns don't accumulate duplicates.
        (d / "text_corpus.txt").write_text("", encoding="utf-8")

    sub_dirs = [d for d in config.SUBS_DIR.iterdir() if d.is_dir()]
    log.info(f"[Corpus] Streaming {len(sub_dirs)} subreddit directories...")

    pool = _HandlePool(config.PROFILE_OPEN_FILE_POOL)
    n_written = 0
    n_records = 0

    try:
        for sub_dir_path in sub_dirs:
            for fname in ("comments.jsonl", "posts.jsonl"):
                fpath = sub_dir_path / fname
                if not fpath.exists():
                    continue
                for rec in iter_jsonl(fpath):
                    n_records += 1
                    author = rec.get("author", "")
                    if author not in qualifying:
                        continue
                    body = (rec.get("body") or "").strip()
                    if not body or body in ("[deleted]", "[removed]"):
                        continue
                    h = pool.get(config.PROFILES_DIR / author / "text_corpus.txt")
                    h.write(body + "\n\n")
                    n_written += 1
                    if n_written % 500_000 == 0:
                        log.info(
                            f"[Corpus] {n_records:,} records scanned, "
                            f"{n_written:,} text snippets written"
                        )
            log.info(f"[Corpus] finished r/{sub_dir_path.name}")
    finally:
        pool.close_all()

    log.info(
        f"[Corpus] DONE — scanned {n_records:,} records, "
        f"wrote {n_written:,} text snippets"
    )


# 5. Per-user profile/activity/index emit

def emit_profiles(activity: dict[str, dict[str, dict]]) -> list[dict]:
    """Walk profile dirs, write activity.json/profile.json, return index list."""
    index = []
    n = 0
    for username, sub_activity in activity.items():
        user_dir = config.PROFILES_DIR / username
        corpus_path = user_dir / "text_corpus.txt"
        if not corpus_path.exists():
            # No text was emitted for them — skip (likely all bodies removed).
            continue

        total_chars = corpus_path.stat().st_size
        # Fast item count via newline pairs (we used "\n\n" separator).
        with open(corpus_path, encoding="utf-8") as f:
            text = f.read()
        n_items = sum(1 for chunk in text.split("\n\n") if chunk.strip())
        avg_chars = round(total_chars / n_items, 1) if n_items else 0

        all_utcs = (
            [r["first_utc"] for r in sub_activity.values() if r["first_utc"] != float("inf")]
            + [r["last_utc"]  for r in sub_activity.values() if r["last_utc"]  != float("-inf")]
        )
        first_utc = min(all_utcs) if all_utcs else None
        last_utc  = max(all_utcs) if all_utcs else None
        span_days = round((last_utc - first_utc) / 86_400, 1) if (first_utc and last_utc) else 0

        write_json(user_dir / "activity.json", {
            sub: {
                "posts":     v["posts"],
                "comments":  v["comments"],
                "first_utc": v["first_utc"] if v["first_utc"] != float("inf") else None,
                "last_utc":  v["last_utc"]  if v["last_utc"]  != float("-inf") else None,
            }
            for sub, v in sub_activity.items()
        })

        profile = {
            "username":           username,
            "n_subreddits":       len(sub_activity),
            "subreddits":         sorted(sub_activity.keys()),
            "total_posts":        sum(r["posts"] for r in sub_activity.values()),
            "total_comments":     sum(r["comments"] for r in sub_activity.values()),
            "total_text_items":   n_items,
            "total_chars":        total_chars,
            "avg_chars_per_item": avg_chars,
            "first_seen_utc":     first_utc,
            "last_seen_utc":      last_utc,
            "temporal_span_days": span_days,
            "built_at":           datetime.now(timezone.utc).isoformat(),
        }
        write_json(user_dir / "profile.json", profile)
        index.append(profile)

        n += 1
        if n % 5000 == 0:
            log.info(f"[Profiles] emitted {n:,}")

    index.sort(key=lambda p: p["total_comments"], reverse=True)
    return index

# main

def main():
    ensure_dirs()

    activity = aggregate_user_activity()
    if not activity:
        return

    qualifying = filter_users(activity)
    if not qualifying:
        log.warning("No users pass thresholds.")
        return

    stream_text_corpora(set(qualifying.keys()))
    index = emit_profiles(qualifying)
    write_json(config.DATA_DIR / "user_index.json", index)

    log.info("=" * 60)
    log.info("STEP 2 COMPLETE")
    log.info(f"  Users written:     {len(index):,}")
    log.info(f"  User index:        {config.DATA_DIR / 'user_index.json'}")
    log.info(f"  Profile directory: {config.PROFILES_DIR}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
