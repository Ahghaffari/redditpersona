"""Disk I/O helpers: JSONL streaming, per-subreddit dirs, checkpoints, anon."""

import hashlib
import hmac
import json
import logging
from pathlib import Path
from typing import Any, Iterator

import config

log = logging.getLogger(__name__)


# directory
def ensure_dirs():
    for d in [
        config.DATA_DIR, config.SUBS_DIR, config.CHECKPOINT_DIR,
        config.LOGS_DIR, config.PROFILES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def sub_dir(sub_name: str) -> Path:
    d = config.SUBS_DIR / sub_name
    d.mkdir(parents=True, exist_ok=True)
    return d

# JSON / JSONL I/O
def append_jsonl(path: Path, record: dict):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def write_json(path: Path, obj: Any):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict]:
    """Materialise a JSONL file into a list. Use iter_jsonl for large files."""
    return list(iter_jsonl(path))


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Stream a JSONL file lazily; tolerates blank or malformed lines."""
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                log.warning(f"Skipping malformed JSONL line in {path}: {e}")

def load_checkpoint(sub_name: str) -> dict:
    path = config.CHECKPOINT_DIR / f"{sub_name}.json"
    if path.exists():
        try:
            return read_json(path)
        except Exception as e:
            log.warning(f"Checkpoint read error for {sub_name}, starting fresh: {e}")
    return {
        "status": "pending", "completed_post_ids": [],
        "posts_collected": 0, "comments_total": 0,
    }


def save_checkpoint(sub_name: str, state: dict):
    write_json(config.CHECKPOINT_DIR / f"{sub_name}.json", state)


def is_sub_complete(sub_name: str) -> bool:
    return load_checkpoint(sub_name).get("status") == "complete"

def anonymize_username(username: str) -> str:
    """Stable, irreversible HMAC-SHA256 hash. Truncated to 16 hex chars."""
    h = hmac.new(
        config.ANON_SALT.encode("utf-8"),
        username.encode("utf-8"),
        hashlib.sha256,
    )
    return "u_" + h.hexdigest()[:16]


def maybe_anon(username: str) -> str:
    """Hash only when ANONYMIZE=True; pass-through otherwise."""
    if not config.ANONYMIZE:
        return username
    if username in config.IGNORED_AUTHORS:
        return username
    return anonymize_username(username)
