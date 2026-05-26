"""
Strategy 3: Semantic embedding grouping (sentence-transformer + K-Means).

Encodes each user's text_corpus.txt with a sentence-transformer, optionally
reduces dimensionality with UMAP, and clusters with K-Means. K is taken
from config or auto-selected by silhouette sweep when not set.

Also writes user_embeddings.npy + user_order.json, reused by Strategy 4 and
by Step 4 analysis for coherence/separation metrics.
"""

import logging
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import ensure_dirs, write_json, read_json

ensure_dirs()
logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

STRATEGY_ID = "strategy_3_semantic"
OUTPUT_DIR  = config.DATA_DIR / "groupings" / STRATEGY_ID


def _load_user_texts(user_index: list[dict]) -> tuple[list[str], list[str]]:
    """
    Load text corpus for each user.
    Returns (usernames, texts) as parallel lists.
    Skips users whose text_corpus.txt is missing or empty.
    """
    usernames, texts = [], []
    for profile in user_index:
        username  = profile["username"]
        corpus_path = config.PROFILES_DIR / username / "text_corpus.txt"
        if not corpus_path.exists():
            continue
        text = corpus_path.read_text(encoding="utf-8").strip()
        if not text:
            continue
        usernames.append(username)
        texts.append(text)
    return usernames, texts


def _encode_texts(texts: list[str], model_name: str) -> np.ndarray:
    """
    Encode a list of user text corpora using a sentence-transformers model.
    Returns L2-normalised numpy array of shape [n_texts, embedding_dim].
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "Missing dependency for Strategy 3.\n"
            "Install with: pip install sentence-transformers"
        )

    log.info(f"[Strategy 3] Loading embedding model: {model_name}")
    hf_token = os.environ.get("HF_TOKEN")
    model = SentenceTransformer(model_name, token=hf_token)
    max_tokens = model.max_seq_length

    # Chunk size in words
    words_per_chunk = int(max_tokens * 0.75)

    def _mean_embed_user(text: str) -> np.ndarray:
        words  = text.split()
        chunks = [
            " ".join(words[i : i + words_per_chunk])
            for i in range(0, max(len(words), 1), words_per_chunk)
        ] or [text]
        chunk_embeddings = model.encode(
            chunks,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            normalize_embeddings=False,
            show_progress_bar=False,
        )
        mean_emb = chunk_embeddings.mean(axis=0)
        norm     = np.linalg.norm(mean_emb)
        return mean_emb / (norm + 1e-8)  # L2-normalise the mean

    log.info(
        f"[Strategy 3] Encoding {len(texts)} user corpora "
        f"(chunk_size≈{words_per_chunk} words, batch_size={config.EMBEDDING_BATCH_SIZE})"
    )
    embeddings = np.stack([_mean_embed_user(t) for t in texts]).astype(np.float32)
    return embeddings


def _select_k(embeddings: np.ndarray) -> int:
    """
    If DEFAULT_N_COMMUNITIES is set, return it directly.
    Otherwise, sweep K values in the 80–150 range and select by silhouette.

    The candidate range is chosen to match the post-hoc consolidated K of
    the graph-based strategies (S2/S4/S5 → ~100), so cross-strategy
    comparisons stay meaningful. We deliberately avoid very small K here:
    silhouette on high-dimensional sentence embeddings is biased toward
    K≈2–5, which previously collapsed S3 to K=5.
    """
    if config.DEFAULT_N_COMMUNITIES is not None:
        return config.DEFAULT_N_COMMUNITIES

    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    k_candidates = [80, 90, 100, 110, 120]
    best_k, best_score = k_candidates[0], -1.0

    for k in k_candidates:
        if k >= len(embeddings):
            continue
        km = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = km.fit_predict(embeddings)
        score  = silhouette_score(
            embeddings, labels,
            metric="cosine",
            sample_size=min(5000, len(embeddings)),
            random_state=42,
        )
        log.info(f"[Strategy 3] K={k:3d}  silhouette={score:.4f}")
        if score > best_score:
            best_k, best_score = k, score

    log.info(f"[Strategy 3] Selected K={best_k} (silhouette={best_score:.4f})")
    return best_k


def run(user_index: list[dict]) -> dict:
    """
    Execute semantic community detection.
    """
    from sklearn.cluster import KMeans

    usernames, texts = _load_user_texts(user_index)
    log.info(f"[Strategy 3] {len(usernames)} users with text corpora")

    embeddings = _encode_texts(texts, config.EMBEDDING_MODEL)

    # Save embeddings for reuse (expensive to recompute)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_DIR / "user_embeddings.npy"), embeddings)
    write_json(OUTPUT_DIR / "user_order.json", usernames)

    k = _select_k(embeddings)
    log.info(f"[Strategy 3] Clustering into K={k} communities (K-Means)")

    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(embeddings)

    # Build community profiles
    community_members: dict[int, list[str]] = defaultdict(list)
    for username, label in zip(usernames, labels):
        community_members[int(label)].append(username)

    community_profiles = []
    for comm_id, members in sorted(community_members.items(), key=lambda x: -len(x[1])):
        # Dominant subreddits in this community
        sub_totals: dict[str, float] = defaultdict(float)
        for user in members:
            activity_path = config.PROFILES_DIR / user / "activity.json"
            if activity_path.exists():
                for sub, rec in read_json(activity_path).items():
                    sub_totals[sub] += rec.get("posts", 0) + rec.get("comments", 0)
        top_subs = sorted(sub_totals.items(), key=lambda x: -x[1])[:10]

        community_profiles.append({
            "community_id": f"sem_{comm_id}",
            "n_members":    len(members),
            "members":      members,
            "top_subreddits": [s for s, _ in top_subs],
        })

    assignments = {
        user: [f"sem_{int(label)}"]
        for user, label in zip(usernames, labels)
    }

    return {
        "assignments": assignments,
        "communities": community_profiles,
        "meta": {
            "strategy":        STRATEGY_ID,
            "algorithm":       "K-Means",
            "embedding_model": config.EMBEDDING_MODEL,
            "k":               k,
            "n_users":         len(usernames),
            "n_communities":   len(community_members),
            "run_at":          datetime.now(timezone.utc).isoformat(),
        },
    }


def main():
    user_index_path = config.DATA_DIR / "user_index.json"
    if not user_index_path.exists():
        log.error("user_index.json not found. Run `python run.py profiles` first.")
        return

    user_index = read_json(user_index_path)
    result = run(user_index)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "community_assignments.json", result["assignments"])
    write_json(OUTPUT_DIR / "community_profiles.json",   result["communities"])
    write_json(OUTPUT_DIR / "grouping_meta.json",        result["meta"])

    log.info(f"[Strategy 3] Done — {result['meta']['n_communities']} communities")


if __name__ == "__main__":
    main()
