"""
dataset.py — Build training samples.

Inputs:
    subreddits/{sub}/posts.jsonl
    subreddits/{sub}/comments.jsonl
    groupings/{strategy_id}/community_assignments.json    # {user: [community_id]}

The dataset is *model-agnostic*: TRL's SFTTrainer detects the `messages`
column and applies the tokenizer's chat template automatically, so the same
data trains correctly on Qwen, Llama-3, Mistral, Gemma, etc.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset

import config
from config import AppConfig, TrainingCfg

logger = logging.getLogger(__name__)

@dataclass
class ThreadSample:
    post_title: str
    post_selftext: str
    ancestor_comments: List[str]
    parent_body: str
    reply_body: str
    reply_author: str
    community: str
    strategy: str

def load_assignments(strategy_id: str, base_dir: Path) -> Dict[str, str]:
    """Load `community_assignments.json` and flatten to user → single label.

    For multi-membership strategies we use the first community
    deterministically (alphabetical first), matching the analyzer. Note
    that strategy_1_subreddit uses content-based assignment in
    `build_samples` (each comment goes to its source subreddit), so this
    flattened map is only used for community enumeration and the
    baseline NMI/ARI in analysis, not for filtering S1 training data.
    """
    path = base_dir / "groupings" / strategy_id / "community_assignments.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        raw = json.load(f)
    flat: Dict[str, str] = {}
    for user, val in raw.items():
        if isinstance(val, list):
            flat[user] = sorted(val)[0] if val else "none"
        else:
            flat[user] = str(val)
    return flat


def load_assignments_multi(strategy_id: str, base_dir: Path) -> Dict[str, list]:
    """Load raw `community_assignments.json` preserving multi-membership lists."""
    path = base_dir / "groupings" / strategy_id / "community_assignments.json"
    if not path.exists():
        raise FileNotFoundError(path)
    with open(path) as f:
        raw = json.load(f)
    out: Dict[str, list] = {}
    for user, val in raw.items():
        if isinstance(val, list):
            out[user] = list(val)
        else:
            out[user] = [str(val)]
    return out


def _iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


# Builder

class DatasetBuilder:
    """Build community-conditioned datasets from collected JSONL."""

    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self.tcfg: TrainingCfg = cfg.training
        self.base_dir = Path(cfg.data_dir)
        self.subs_dir = self.base_dir / "subreddits"

        self._posts: Optional[Dict[str, dict]] = None
        self._comments_by_id: Optional[Dict[str, dict]] = None
        self._comments: Optional[List[dict]] = None

    def _ensure_indexes(self):
        if self._comments is not None:
            return
        logger.info("Building post/comment indexes from %s ...", self.subs_dir)
        posts: Dict[str, dict] = {}
        comments: List[dict] = []
        comments_by_id: Dict[str, dict] = {}

        sub_dirs = [d for d in self.subs_dir.iterdir() if d.is_dir()]
        for d in sub_dirs:
            sub_name = d.name
            pp = d / "posts.jsonl"
            if pp.exists():
                for r in _iter_jsonl(pp):
                    r["_subreddit"] = sub_name
                    posts[r["id"]] = r
            cp = d / "comments.jsonl"
            if cp.exists():
                for r in _iter_jsonl(cp):
                    r["_subreddit"] = sub_name
                    comments.append(r)
                    comments_by_id[r["id"]] = r

        self._posts = posts
        self._comments = comments
        self._comments_by_id = comments_by_id
        logger.info("Indexed %d posts, %d comments", len(posts), len(comments))

    # ancestor walk
    def _ancestors(self, comment: dict) -> List[str]:
        chain: List[str] = []
        cur = comment
        while len(chain) < self.tcfg.max_context_comments:
            pid = cur.get("parent_id", "")
            if not pid.startswith("t1_"):
                break
            parent = self._comments_by_id.get(pid[3:])
            if parent is None:
                break
            chain.append(parent.get("body", ""))
            cur = parent
        chain.reverse()
        return chain

    # per-community sample build
    def build_samples(
        self,
        strategy: str,
        community: str,
        assignments: Dict[str, str],
    ) -> List[ThreadSample]:
        self._ensure_indexes()
        is_subreddit_strategy = (strategy == "strategy_1_subreddit")
        if is_subreddit_strategy:
            members = None
        else:
            members = {u for u, c in assignments.items() if c == community}

        out: List[ThreadSample] = []
        skipped_short = skipped_no_parent = 0
        for c in self._comments:
            if is_subreddit_strategy:
                if c.get("_subreddit") != community:
                    continue
            else:
                author = c.get("author", "")
                if author not in members:
                    continue
            pid = c.get("parent_id", "")
            if not pid.startswith("t1_"):
                continue
            body = (c.get("body") or "").strip()
            if len(body) < self.tcfg.min_reply_length:
                skipped_short += 1
                continue
            parent = self._comments_by_id.get(pid[3:])
            if parent is None:
                skipped_no_parent += 1
                continue
            post = self._posts.get(c.get("post_id", ""))
            if post is None:
                continue
            out.append(ThreadSample(
                post_title=post.get("title", ""),
                post_selftext=post.get("body", ""),
                ancestor_comments=self._ancestors(c),
                parent_body=parent.get("body", ""),
                reply_body=body,
                reply_author=c.get("author", ""),
                community=community,
                strategy=strategy,
            ))
        logger.info(
            "strategy=%s community=%s: %d samples (skipped %d short, %d no parent)",
            strategy, community, len(out), skipped_short, skipped_no_parent,
        )
        return out

    # Messages formatting (model-agnostic)
    @staticmethod
    def to_messages(s: ThreadSample) -> List[Dict[str, str]]:
        """Render a sample as a list of role/content messages.

        The tokenizer's chat template (Qwen ChatML, Llama-3 headers, Mistral
        [INST], Gemma turns, …) is applied at training/eval time — this
        function only assembles the *content*.
        """
        ctx_lines = [f"Post: {s.post_title}"]
        if s.post_selftext:
            ctx_lines.append(s.post_selftext)
        if s.ancestor_comments:
            ctx_lines.append("Thread context:")
            for i, a in enumerate(s.ancestor_comments, 1):
                ctx_lines.append(f"  [{i}] {a}")
        ctx_lines.append(f"Parent comment: {s.parent_body}")
        return [
            {
                "role": "system",
                "content": (
                    f"You are a Reddit user from community {s.community} "
                    f"(grouped by {s.strategy}). Reply naturally to the parent "
                    f"comment in the thread context."
                ),
            },
            {"role": "user", "content": "\n".join(ctx_lines)},
            {"role": "assistant", "content": s.reply_body},
        ]

    # HuggingFace splits
    def _splits(
        self, samples: List[ThreadSample], strategy: str, community: str,
    ) -> Optional[Dict[str, Dataset]]:
        if len(samples) < 10:
            logger.warning("strategy=%s community=%s: only %d samples — skip",
                           strategy, community, len(samples))
            return None
        method = getattr(self.tcfg, "split_method", "random")
        if method == "user":
            return self._splits_by_user(samples, strategy, community)
        if method != "random":
            raise ValueError(
                f"Unknown split_method={method!r} (expected 'random' or 'user')"
            )
        messages = [self.to_messages(s) for s in samples]
        seed = int(hashlib.sha256(f"{strategy}_{community}".encode()).hexdigest()[:8], 16)
        ds = Dataset.from_dict({"messages": messages}).shuffle(seed=seed)
        n = len(ds)
        n_test = max(1, int(n * self.tcfg.test_ratio))
        n_val  = max(1, int(n * self.tcfg.val_ratio))
        n_train = n - n_test - n_val
        if n_train < 5:
            return None
        cap = getattr(self.tcfg, "max_train_samples", 0)
        if cap and cap > 0 and n_train > cap:
            logger.info(
                "strategy=%s community=%s: capping train %d → %d",
                strategy, community, n_train, cap,
            )
            n_train = cap
        return {
            "train": ds.select(range(n_train)),
            "val":   ds.select(range(n_train, n_train + n_val)),
            "test":  ds.select(range(n_train + n_val, n)),
        }

    def _splits_by_user(
        self, samples: List[ThreadSample], strategy: str, community: str,
    ) -> Optional[Dict[str, Dataset]]:
        seed_bytes = hashlib.sha256(f"{strategy}_{community}_user".encode()).digest()
        salt = seed_bytes.hex()[:16]
        authors = sorted({s.reply_author for s in samples if s.reply_author})
        if not authors:
            return None
        test_ratio = self.tcfg.test_ratio
        val_ratio = self.tcfg.val_ratio
        train_authors, val_authors, test_authors = set(), set(), set()
        for a in authors:
            h = int(hashlib.sha256(f"{salt}_{a}".encode()).hexdigest()[:8], 16)
            r = (h % 10_000) / 10_000.0
            if r < test_ratio:
                test_authors.add(a)
            elif r < test_ratio + val_ratio:
                val_authors.add(a)
            else:
                train_authors.add(a)
        train_samples, val_samples, test_samples = [], [], []
        for s in samples:
            if s.reply_author in test_authors:
                test_samples.append(s)
            elif s.reply_author in val_authors:
                val_samples.append(s)
            else:
                train_samples.append(s)
        if len(train_samples) < 5 or not test_samples or not val_samples:
            logger.warning(
                "strategy=%s community=%s: user-split too small "
                "(train=%d val=%d test=%d) — skip",
                strategy, community,
                len(train_samples), len(val_samples), len(test_samples),
            )
            return None
        cap = getattr(self.tcfg, "max_train_samples", 0)
        if cap and cap > 0 and len(train_samples) > cap:
            logger.info(
                "strategy=%s community=%s: capping train %d → %d",
                strategy, community, len(train_samples), cap,
            )
            shuffle_seed = int(hashlib.sha256(f"{salt}_train".encode()).hexdigest()[:8], 16)
            import random as _random
            rng = _random.Random(shuffle_seed)
            rng.shuffle(train_samples)
            train_samples = train_samples[:cap]
        def _to_ds(rows: List[ThreadSample], shuffle_tag: str) -> Dataset:
            messages = [self.to_messages(s) for s in rows]
            shuffle_seed = int(
                hashlib.sha256(f"{salt}_{shuffle_tag}".encode()).hexdigest()[:8], 16
            )
            return Dataset.from_dict({"messages": messages}).shuffle(seed=shuffle_seed)
        return {
            "train": _to_ds(train_samples, "train"),
            "val":   _to_ds(val_samples, "val"),
            "test":  _to_ds(test_samples, "test"),
        }

    def build_dataset(
        self, strategy: str, community: str, assignments: Dict[str, str],
    ) -> Optional[Dict[str, Dataset]]:
        return self._splits(self.build_samples(strategy, community, assignments),
                            strategy, community)

    def build_strategy_dataset(
        self, strategy: str, assignments: Dict[str, str], max_communities: int = 0,
    ) -> Optional[Dict[str, Dataset]]:
        """Pool all communities of one strategy into a single dataset.

        The community label remains in the system prompt so the model
        learns to condition on it. For strategy_1_subreddit, communities are
        the per-subreddit buckets (each comment assigned by source subreddit).
        """
        from collections import Counter
        if strategy == "strategy_1_subreddit":
            self._ensure_indexes()
            counts = Counter(c.get("_subreddit") for c in self._comments
                             if c.get("_subreddit"))
        else:
            counts = Counter(assignments.values())
        comms = [c for c, _ in counts.most_common()]
        if max_communities > 0:
            comms = comms[:max_communities]
        all_samples: List[ThreadSample] = []
        for comm in comms:
            all_samples.extend(self.build_samples(strategy, comm, assignments))
        return self._splits(all_samples, strategy, "pooled")

    def build_baseline_dataset(self) -> Optional[Dict[str, Dataset]]:
        """All replies, single 'all' community label (no grouping)."""
        self._ensure_indexes()
        samples: List[ThreadSample] = []
        for c in self._comments:
            pid = c.get("parent_id", "")
            if not pid.startswith("t1_"):
                continue
            body = (c.get("body") or "").strip()
            if len(body) < self.tcfg.min_reply_length:
                continue
            parent = self._comments_by_id.get(pid[3:])
            if parent is None:
                continue
            post = self._posts.get(c.get("post_id", ""))
            if post is None:
                continue
            samples.append(ThreadSample(
                post_title=post.get("title", ""),
                post_selftext=post.get("body", ""),
                ancestor_comments=self._ancestors(c),
                parent_body=parent.get("body", ""),
                reply_body=body,
                reply_author=c.get("author", ""),
                community="all",
                strategy="baseline_all",
            ))
        logger.info("baseline_all: %d samples", len(samples))
        return self._splits(samples, "baseline_all", "all")
