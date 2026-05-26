"""
Step 1: Collect posts, comments, and interactions via AsyncPRAW.

For each verified subreddit writes:
    data/subreddits/{name}/{posts,comments}.jsonl + meta.json
and streams two global files during collection:
    data/user_activity_matrix.jsonl   (user × subreddit counts)
    data/user_interactions.jsonl      (directed reply edges)

Resumable: per-subreddit checkpoints under data/checkpoints/.
"""

import asyncio
import json
import logging
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import asyncpraw
import asyncprawcore

import config
from utils import (
    build_async_reddit,
    ensure_dirs,
    sub_dir,
    append_jsonl,
    write_json,
    read_json,
    load_checkpoint,
    save_checkpoint,
    maybe_anon,
)

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
        logging.FileHandler(config.LOGS_DIR / "collect.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def serialize_post(submission, sub_name: str, sort_name: str) -> dict:
    author = str(submission.author) if submission.author else "[deleted]"
    return {
        "id":                 submission.id,
        "subreddit":          sub_name,
        "title":              submission.title,
        "body":               submission.selftext if submission.is_self else "",
        "author":             maybe_anon(author),
        "score":              submission.score,
        "upvote_ratio":       getattr(submission, "upvote_ratio", None),
        "num_comments":       submission.num_comments,
        "created_utc":        submission.created_utc,
        "url":                submission.url,
        "permalink":          submission.permalink,
        "flair":              submission.link_flair_text,
        "is_self":            submission.is_self,
        "is_stickied":        submission.stickied,
        "sort_discovered_in": sort_name,
        "collected_at":       _ts(),
    }


def serialize_comment(
    comment,
    sub_name: str,
    post_id: str,
    post_author: str,
) -> dict:
    author = str(comment.author) if comment.author else "[deleted]"
    parent_id   = comment.parent_id           # "t1_xxx" or "t3_xxx"
    parent_type = "post" if parent_id.startswith("t3_") else "comment"
    return {
        "id":           comment.id,
        "subreddit":    sub_name,
        "post_id":      post_id,
        "post_author":  maybe_anon(post_author),
        "body":         comment.body,
        "author":       maybe_anon(author),
        "score":        comment.score,
        "created_utc":  comment.created_utc,
        "parent_id":    parent_id,
        "parent_type":  parent_type,
        "depth":        getattr(comment, "depth", None),
        "permalink":    comment.permalink,
        "is_submitter": comment.is_submitter,
        "collected_at": _ts(),
    }

class ActivityTracker:
    """
    Accumulates user-activity and reply-interaction data in memory during
    collection of one subreddit, then flushes to global JSONL files.

    Keeps (user, subreddit) aggregates so that multiple flushes for the same
    subreddit are additive, not duplicated.
    """

    def __init__(self):
        # (user, subreddit) → {"posts": int, "comments": int,
        #                       "first_utc": float, "last_utc": float}
        self._activity: dict[tuple[str, str], dict] = {}
        # (from_user, to_user, subreddit) → count
        self._interactions: dict[tuple[str, str, str], int] = defaultdict(int)

    def _valid_author(self, author: str) -> bool:
        return author not in config.IGNORED_AUTHORS

    def record_post(self, author: str, subreddit: str, utc: float):
        if not self._valid_author(author):
            return
        key = (author, subreddit)
        rec = self._activity.setdefault(key, {
            "posts": 0, "comments": 0, "first_utc": utc, "last_utc": utc
        })
        rec["posts"] += 1
        rec["first_utc"] = min(rec["first_utc"], utc)
        rec["last_utc"]  = max(rec["last_utc"],  utc)

    def record_comment(self, author: str, subreddit: str, utc: float):
        if not self._valid_author(author):
            return
        key = (author, subreddit)
        rec = self._activity.setdefault(key, {
            "posts": 0, "comments": 0, "first_utc": utc, "last_utc": utc
        })
        rec["comments"] += 1
        rec["first_utc"] = min(rec["first_utc"], utc)
        rec["last_utc"]  = max(rec["last_utc"],  utc)

    def record_reply(self, from_author: str, to_author: str, subreddit: str):
        if (
            self._valid_author(from_author)
            and self._valid_author(to_author)
            and from_author != to_author
        ):
            self._interactions[(from_author, to_author, subreddit)] += 1

    def flush(self):
        """Write accumulated data to global JSONL files and clear buffers."""
        if config.BUILD_USER_ACTIVITY_MATRIX:
            for (user, subreddit), rec in self._activity.items():
                append_jsonl(config.USER_MATRIX_FILE, {
                    "user":           user,
                    "subreddit":      subreddit,
                    "posts":          rec["posts"],
                    "comments":       rec["comments"],
                    "first_seen_utc": rec["first_utc"],
                    "last_seen_utc":  rec["last_utc"],
                })
            self._activity.clear()

        if config.BUILD_INTERACTION_GRAPH:
            for (frm, to, sub), weight in self._interactions.items():
                append_jsonl(config.INTERACTION_FILE, {
                    "from":      frm,
                    "to":        to,
                    "subreddit": sub,
                    "weight":    weight,
                })
            self._interactions.clear()


async def expand_and_collect_comments(
    submission,
    sub_name: str,
    tracker: ActivityTracker,
    semaphore: asyncio.Semaphore,
) -> list[dict]:
    """
    Expand the comment tree for one submission, serialize all valid comments,
    record activity + interactions. Returns list of serialised comment dicts.

    The semaphore limits concurrent replace_more() calls across a batch of
    posts to avoid saturating the rate limit.
    """
    post_id     = submission.id
    post_author = str(submission.author) if submission.author else "[deleted]"

    async with semaphore:
        try:
            await submission.load()
        except Exception as exc:
            log.warning(f"[{sub_name}] Failed to load post {post_id}: {exc} — skipping comments")
            return []

        try:
            await asyncio.wait_for(
                submission.comments.replace_more(limit=config.REPLACE_MORE_LIMIT),
                timeout=config.REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            log.warning(
                f"[{sub_name}] replace_more timeout for post {post_id} — using partial tree"
            )
        except Exception as exc:
            log.warning(
                f"[{sub_name}] replace_more error for post {post_id}: {exc} — using partial"
            )

    all_comments = submission.comments.list()

    parent_authors: dict[str, str] = {f"t3_{post_id}": post_author}

    serialised: list[dict] = []
    for comment in all_comments:
        if not hasattr(comment, "body"):
            continue  # MoreComments remnant
        body = comment.body
        if not body or body in ("[deleted]", "[removed]"):
            continue
        if len(body) < config.COMMENT_MIN_BODY_CHARS:
            continue

        c_author = str(comment.author) if comment.author else "[deleted]"
        rec = serialize_comment(comment, sub_name, post_id, post_author)
        serialised.append(rec)

        utc = comment.created_utc
        tracker.record_comment(maybe_anon(c_author), sub_name, utc)

        # Register this comment in the parent lookup for deeper replies
        parent_authors[f"t1_{comment.id}"] = c_author

        # Interaction: commenter replied to whoever authored the parent
        if config.BUILD_INTERACTION_GRAPH:
            p_author = parent_authors.get(comment.parent_id)
            if p_author and p_author not in config.IGNORED_AUTHORS:
                tracker.record_reply(
                    maybe_anon(c_author),
                    maybe_anon(p_author),
                    sub_name,
                )

    return serialised


async def collect_subreddit(
    reddit: asyncpraw.Reddit,
    sub_name: str,
    tracker: ActivityTracker,
) -> dict:
    """
    Collect all posts + comment trees for one subreddit.

    Strategy:
      1. Fetch posts via each configured sort order (hot / top / new).
      2. Process posts in batches; within each batch expand comment trees
         concurrently (bounded by MAX_CONCURRENT_REQUESTS semaphore).
      3. Write posts and comments incrementally to JSONL.
      4. Checkpoint every CHECKPOINT_EVERY_N_POSTS posts.
    """
    out_dir       = sub_dir(sub_name)
    posts_path    = out_dir / "posts.jsonl"
    comments_path = out_dir / "comments.jsonl"
    meta_path     = out_dir / "meta.json"

    # Resume
    ck = load_checkpoint(sub_name)
    if ck.get("status") == "complete":
        log.info(f"[{sub_name}] Already complete — skipping")
        return ck

    completed_ids: set[str] = set(ck.get("completed_post_ids", []))
    posts_done    = ck.get("posts_collected", 0)
    comments_done = ck.get("comments_total",  0)
    t0 = time.time()

    log.info(
        f"[{sub_name}] Starting (resume: {posts_done} posts already done)"
    )

    try:
        subreddit = await reddit.subreddit(sub_name)

        post_objects: dict[str, object] = {}
        ordered_ids:  list[str]         = []

        for sort in config.POST_SORTS:
            log.info(f"[{sub_name}] Fetching post list ({sort})")
            try:
                if sort == "top":
                    listing = subreddit.top(
                        time_filter=config.TOP_TIME_FILTER,
                        limit=config.POSTS_PER_SUBREDDIT,
                    )
                elif sort == "controversial":
                    listing = subreddit.controversial(
                        time_filter=config.TOP_TIME_FILTER,
                        limit=config.POSTS_PER_SUBREDDIT,
                    )
                elif sort == "new":
                    listing = subreddit.new(limit=config.POSTS_PER_SUBREDDIT)
                else:
                    listing = subreddit.hot(limit=config.POSTS_PER_SUBREDDIT)

                async for submission in listing:
                    if submission.id not in post_objects:
                        post_objects[submission.id] = submission
                        ordered_ids.append(submission.id)

                await asyncio.sleep(config.SLEEP_BETWEEN_POSTS)

            except asyncprawcore.exceptions.NotFound:
                log.warning(f"[{sub_name}] 404 on sort={sort}, skipping sort")
            except Exception as exc:
                log.warning(f"[{sub_name}] Error fetching {sort} list: {exc}")

        new_ids = [pid for pid in ordered_ids if pid not in completed_ids]
        log.info(
            f"[{sub_name}] {len(post_objects)} unique posts discovered, "
            f"{len(new_ids)} new to collect"
        )

        semaphore = asyncio.Semaphore(config.MAX_CONCURRENT_REQUESTS)
        batch_size = config.MAX_CONCURRENT_REQUESTS * 2  # 2× semaphore slots

        for batch_start in range(0, len(new_ids), batch_size):
            batch_ids = new_ids[batch_start : batch_start + batch_size]
            batch     = [post_objects[pid] for pid in batch_ids]

            comment_results = await asyncio.gather(*[
                expand_and_collect_comments(sub, sub_name, tracker, semaphore)
                for sub in batch
            ])

            for submission, comments in zip(batch, comment_results):
                post_author = str(submission.author) if submission.author else "[deleted]"
                post_data   = serialize_post(submission, sub_name, "batch")
                append_jsonl(posts_path, post_data)

                utc = submission.created_utc
                tracker.record_post(maybe_anon(post_author), sub_name, utc)

                for cdata in comments:
                    append_jsonl(comments_path, cdata)

                completed_ids.add(submission.id)
                posts_done    += 1
                comments_done += len(comments)

            # Checkpoint after each batch
            if posts_done % config.CHECKPOINT_EVERY_N_POSTS < batch_size:
                ck_state = {
                    "status":             "in_progress",
                    "completed_post_ids": list(completed_ids),
                    "posts_collected":    posts_done,
                    "comments_total":     comments_done,
                }
                save_checkpoint(sub_name, ck_state)
                tracker.flush()
                log.info(
                    f"[{sub_name}] checkpoint: {posts_done} posts, "
                    f"{comments_done} comments"
                )

            await asyncio.sleep(config.SLEEP_BETWEEN_POSTS * batch_size)

        elapsed = round(time.time() - t0, 1)
        final   = {
            "status":             "complete",
            "completed_post_ids": list(completed_ids),
            "posts_collected":    posts_done,
            "comments_total":     comments_done,
            "elapsed_seconds":    elapsed,
            "completed_at":       _ts(),
        }
        save_checkpoint(sub_name, final)
        write_json(meta_path, final)
        tracker.flush()

        log.info(
            f"[{sub_name}] DONE — {posts_done} posts, "
            f"{comments_done} comments in {elapsed}s"
        )
        return final

    except asyncprawcore.exceptions.NotFound:
        err = {"status": "failed", "reason": "not_found"}
        save_checkpoint(sub_name, err)
        log.error(f"[{sub_name}] Subreddit not found (deleted after verification?)")
        return err

    except asyncprawcore.exceptions.Forbidden:
        err = {"status": "failed", "reason": "forbidden"}
        save_checkpoint(sub_name, err)
        log.error(f"[{sub_name}] Forbidden — subreddit became private")
        return err

    except Exception as exc:
        err = {
            "status":          "failed",
            "reason":          f"{type(exc).__name__}: {exc}",
            "posts_collected": posts_done,
            "comments_total":  comments_done,
        }
        save_checkpoint(sub_name, err)
        log.error(f"[{sub_name}] Unexpected error: {exc}")
        return err


async def main():
    ensure_dirs()

    if not config.VERIFIED_SUBS_FILE.exists():
        log.error(
            "verified_subreddits.json not found.\n"
            "Run `python run.py verify` first."
        )
        return

    with open(config.VERIFIED_SUBS_FILE, encoding="utf-8") as f:
        all_verified = json.load(f)

    target_subs = [r["name"] for r in all_verified if r.get("verified")]
    if not target_subs:
        log.error("No verified subreddits found. Re-run `python run.py verify`.")
        return

    log.info("=" * 60)
    log.info(f"STEP 1: Collecting data from {len(target_subs)} verified subreddits")
    log.info("=" * 60)

    reddit  = build_async_reddit()
    tracker = ActivityTracker()
    results = []

    for idx, sub_name in enumerate(target_subs, start=1):
        log.info(f"\n[{idx}/{len(target_subs)}] r/{sub_name}")
        result = await collect_subreddit(reddit, sub_name, tracker)
        results.append({"subreddit": sub_name, **result})
        await asyncio.sleep(config.SLEEP_BETWEEN_SUBS)

    await reddit.close()
    tracker.flush()  # Final flush for any remaining buffered data

    n_complete = sum(1 for r in results if r.get("status") == "complete")
    n_failed   = sum(1 for r in results if r.get("status") == "failed")
    total_posts    = sum(r.get("posts_collected", 0) for r in results)
    total_comments = sum(r.get("comments_total",  0) for r in results)

    summary = {
        "collected_at":         _ts(),
        "subreddits_complete":  n_complete,
        "subreddits_failed":    n_failed,
        "total_posts":          total_posts,
        "total_comments":       total_comments,
        "results":              results,
    }
    write_json(config.COLLECTION_SUMMARY, summary)

    log.info("")
    log.info("=" * 60)
    log.info("COLLECTION COMPLETE")
    log.info(f"  Subreddits:      {n_complete} complete, {n_failed} failed")
    log.info(f"  Total posts:     {total_posts:,}")
    log.info(f"  Total comments:  {total_comments:,}")
    log.info(f"  Summary:         {config.COLLECTION_SUMMARY}")
    log.info("=" * 60)
    log.info("Next step: python run.py profiles")


if __name__ == "__main__":
    asyncio.run(main())
