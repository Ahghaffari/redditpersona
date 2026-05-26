"""AsyncPRAW client factory + retry helper."""

import asyncio
import logging

import asyncpraw
import asyncprawcore

import config

log = logging.getLogger(__name__)


def build_async_reddit() -> asyncpraw.Reddit:
    """Configured AsyncPRAW client. Caller must `await reddit.close()`."""
    if not config.REDDIT_CLIENT_ID or not config.REDDIT_CLIENT_SECRET:
        raise EnvironmentError(
            "REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET must be set "
            "(in .env or environment)."
        )
    return asyncpraw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent=config.REDDIT_USER_AGENT,
        ratelimit_seconds=config.RATELIMIT_SECONDS,
        timeout=int(config.REQUEST_TIMEOUT),
    )


async def fetch_with_retry(coro_factory, label: str = "request"):
    """Exponential-backoff retry for a coroutine factory."""
    for attempt in range(config.MAX_RETRIES):
        try:
            return await asyncio.wait_for(
                coro_factory(), timeout=config.REQUEST_TIMEOUT,
            )
        except asyncio.TimeoutError:
            wait = config.BACKOFF_BASE ** attempt
            log.warning(f"[{label}] Timeout (attempt {attempt + 1}), retry in {wait:.1f}s")
            await asyncio.sleep(wait)
        except asyncprawcore.exceptions.TooManyRequests:
            wait = config.BACKOFF_BASE ** (attempt + 2)
            log.warning(f"[{label}] Rate limited (429), retry in {wait:.1f}s")
            await asyncio.sleep(wait)
        except asyncprawcore.exceptions.ServerError as exc:
            wait = config.BACKOFF_BASE ** attempt
            log.warning(f"[{label}] Server error ({exc}), retry in {wait:.1f}s")
            await asyncio.sleep(wait)
        except (asyncprawcore.exceptions.NotFound,
                asyncprawcore.exceptions.Forbidden,
                asyncprawcore.exceptions.Redirect):
            return None
        except Exception as exc:
            if attempt == config.MAX_RETRIES - 1:
                log.error(f"[{label}] All {config.MAX_RETRIES} attempts failed: {exc}")
                raise
            wait = config.BACKOFF_BASE ** attempt
            log.warning(
                f"[{label}] {type(exc).__name__} on attempt {attempt + 1}, "
                f"retry in {wait:.1f}s: {exc}"
            )
            await asyncio.sleep(wait)
    return None
