"""
Step 0: Verify subreddits exist and are collectable.

For each entry in config.ALL_SUBREDDITS makes one /r/{name}/about call and
keeps subreddits that are: existing, public, SFW (if REQUIRE_SFW), and have
>= MIN_SUBSCRIBERS subscribers. Result → data/verified_subreddits.json,
consumed by Step 1.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone

import asyncpraw
import asyncprawcore

import config
from utils import ensure_dirs, write_json

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
        logging.FileHandler(config.LOGS_DIR / "verify.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)


async def verify_one(
    reddit: asyncpraw.Reddit,
    sub_name: str,
    semaphore: asyncio.Semaphore,
) -> dict:
    """
    Fetch subreddit metadata and run all checks.
    Returns a result dict with 'verified' (bool) and 'reason' (str | None).
    """
    result = {
        "name":              sub_name,
        "category":          config.SUB_TO_CATEGORY.get(sub_name, "unknown"),
        "verified":          False,
        "reason":            None,
        "subscribers":       None,
        "active_user_count": None,
        "over18":            None,
        "subreddit_type":    None,
        "description":       None,
        "created_utc":       None,
        "checked_at":        datetime.now(timezone.utc).isoformat(),
    }

    async with semaphore:
        try:
            # fetch=True forces an immediate API call to load all attributes
            sub = await reddit.subreddit(sub_name, fetch=True)

            result["subscribers"]       = getattr(sub, "subscribers",       None)
            result["active_user_count"] = getattr(sub, "active_user_count", None)
            result["over18"]            = getattr(sub, "over18",            None)
            result["subreddit_type"]    = getattr(sub, "subreddit_type",    None)
            result["created_utc"]       = getattr(sub, "created_utc",       None)
            desc = getattr(sub, "public_description", "") or ""
            result["description"] = desc[:300]

        except asyncprawcore.exceptions.NotFound:
            result["reason"] = "not_found (404)"
            return result
        except asyncprawcore.exceptions.Forbidden:
            result["reason"] = "forbidden (403) — private or quarantined"
            return result
        except asyncprawcore.exceptions.Redirect:
            result["reason"] = "redirect — subreddit may have been renamed"
            return result
        except asyncprawcore.exceptions.TooManyRequests:
            result["reason"] = "rate_limited — retry later"
            return result
        except Exception as exc:
            result["reason"] = f"error: {type(exc).__name__}: {exc}"
            return result
        finally:
            # Small sleep inside semaphore to spread requests evenly
            await asyncio.sleep(0.4)

    if config.REQUIRE_PUBLIC and result["subreddit_type"] != "public":
        result["reason"] = f"not_public (type={result['subreddit_type']})"
        return result

    if config.REQUIRE_SFW and result["over18"]:
        result["reason"] = "nsfw (over18=True)"
        return result

    subscribers = result["subscribers"] or 0
    if subscribers < config.MIN_SUBSCRIBERS:
        result["reason"] = f"too_small ({subscribers:,} < {config.MIN_SUBSCRIBERS:,} subscribers)"
        return result

    result["verified"] = True
    return result


async def main():
    ensure_dirs()

    if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
        log.error("Reddit credentials not set. Check your .env file.")
        return

    reddit = asyncpraw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
        ratelimit_seconds=60,
        timeout=15,
    )

    # Limit to 3 concurrent checks
    semaphore = asyncio.Semaphore(3)

    log.info("=" * 60)
    log.info(f"Verifying {len(config.ALL_SUBREDDITS)} candidate subreddits")
    log.info("=" * 60)

    tasks = [verify_one(reddit, sub, semaphore) for sub in config.ALL_SUBREDDITS]
    results: list[dict] = await asyncio.gather(*tasks)

    await reddit.close()

    passed = [r for r in results if r["verified"]]
    failed = [r for r in results if not r["verified"]]

    results.sort(key=lambda r: (not r["verified"], r["name"].lower()))
    write_json(config.VERIFIED_SUBS_FILE, results)

    log.info("")
    log.info(f"Passed: {len(passed)} / {len(results)}")
    log.info(f"Failed: {len(failed)}")
    log.info("")

    # Per-category breakdown
    from collections import defaultdict
    cat_pass: dict[str, list[str]] = defaultdict(list)
    cat_fail: dict[str, list[str]] = defaultdict(list)
    for r in results:
        cat = r["category"]
        if r["verified"]:
            cat_pass[cat].append(r["name"])
        else:
            cat_fail[cat].append(r["name"])

    for cat in sorted(config.SUBREDDITS.keys()):
        total = len(config.SUBREDDITS[cat])
        ok    = len(cat_pass.get(cat, []))
        bad   = cat_fail.get(cat, [])
        log.info(f"  [{cat:<25}]  {ok}/{total} passed")
        if bad:
            reasons = {r["name"]: r["reason"] for r in failed if r["name"] in bad}
            for name, reason in reasons.items():
                log.info(f"      ✗ {name:<30} → {reason}")

    log.info("")
    log.info(f"Verified subreddit list saved to: {config.VERIFIED_SUBS_FILE}")
    log.info("Run `python run.py collect` to begin collection.")


if __name__ == "__main__":
    asyncio.run(main())
