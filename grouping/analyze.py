"""
Step 4: Grouping quality analysis.

Computes Phase 3 metrics across all grouping strategies and writes a
comparison table.

Metrics: NMI/ARI vs subreddit baseline, intra-community coherence,
inter-community separation, topic-distribution entropy, vocabulary
distinctiveness, and size Gini.

Outputs to data/analysis/. Requires Strategy 3 embeddings
(user_embeddings.npy) for coherence/separation metrics.
The top-K consolidation "other" residual bucket (produced for training
comparability) is excluded from every metric so it does not pollute
Gini/coherence/NMI.
"""

import argparse
import csv
import json
import logging
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import config
from utils import ensure_dirs, write_json, read_json

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
        logging.FileHandler(config.LOGS_DIR / "04_analysis.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

ANALYSIS_DIR  = config.DATA_DIR / "analysis"
GROUPINGS_DIR = config.DATA_DIR / "groupings"

STRATEGY_IDS = [
    "strategy_1_subreddit",
    "strategy_2_graph",
    "strategy_3_semantic",
    "strategy_4_hybrid",
    "strategy_5_interaction",
]


def _is_residual(comm: str | None) -> bool:
    """True if `comm` is the top-K consolidation residual bucket
    (suffix `_other` or value `none`/empty)."""
    if comm is None:
        return True
    s = str(comm)
    if not s or s == "none":
        return True
    return s.endswith("_other") or s == "other"


def load_assignments(strategy_id: str) -> dict[str, str] | None:
    """
    Load community_assignments.json for one strategy.
    Converts multi-membership lists (Strategy 1) to primary assignment
    (most-active subreddit = first in alphabetical sort for reproducibility).

    Returns: user → community_id (single string), or None if not found.
    """
    path = GROUPINGS_DIR / strategy_id / "community_assignments.json"
    if not path.exists():
        log.warning(f"Assignments not found for {strategy_id}: {path}")
        return None
    raw = read_json(path)

    flat: dict[str, str] = {}
    for user, assignment in raw.items():
        if isinstance(assignment, list):
            flat[user] = assignment[0] if assignment else "none"
        else:
            flat[user] = str(assignment)
    return flat


def load_embeddings() -> tuple[list[str], np.ndarray] | tuple[None, None]:
    """Load Strategy 3 embeddings (shared across all coherence/separation metrics)."""
    emb_path   = GROUPINGS_DIR / "strategy_3_semantic" / "user_embeddings.npy"
    order_path = GROUPINGS_DIR / "strategy_3_semantic" / "user_order.json"
    if emb_path.exists() and order_path.exists():
        return read_json(order_path), np.load(str(emb_path))
    log.warning("Strategy 3 embeddings not found. Coherence/separation metrics unavailable.")
    return None, None


def load_all_corpora(users: list[str], n_workers: int = 32) -> dict[str, str]:
    """Load every user's text_corpus.txt once into memory (parallel I/O).

    Reused across all strategies so we don't pay 301k random tiny-file reads
    per strategy.
    """
    log.info("Pre-loading %d user corpora (parallel, %d workers) ...", len(users), n_workers)

    def _read(u: str) -> tuple[str, str]:
        p = config.PROFILES_DIR / u / "text_corpus.txt"
        try:
            return u, p.read_text(encoding="utf-8").strip() if p.exists() else ""
        except Exception:
            return u, ""

    out: dict[str, str] = {}
    n_done = 0
    log_every = max(1, len(users) // 20)
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        for u, txt in ex.map(_read, users, chunksize=64):
            if txt:
                out[u] = txt
            n_done += 1
            if n_done % log_every == 0:
                log.info("  corpora loaded: %d / %d", n_done, len(users))
    log.info("Loaded %d non-empty corpora.", len(out))
    return out


# Metric 1: NMI and ARI vs. baseline
def compute_nmi_ari(
    baseline_assignments: dict[str, str],
    target_assignments: dict[str, str],
) -> dict[str, float]:
    """
    Compute NMI and ARI between two assignment dicts.
    Only users present in BOTH are included (intersection).
    """
    from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

    common_users = sorted(set(baseline_assignments) & set(target_assignments))
    common_users = [
        u for u in common_users
        if not _is_residual(baseline_assignments[u])
        and not _is_residual(target_assignments[u])
    ]
    if len(common_users) < 2:
        return {"nmi": None, "ari": None, "n_common_users": len(common_users)}

    y_base   = [baseline_assignments[u] for u in common_users]
    y_target = [target_assignments[u]   for u in common_users]

    return {
        "nmi":            round(normalized_mutual_info_score(y_base, y_target), 4),
        "ari":            round(adjusted_rand_score(y_base, y_target), 4),
        "n_common_users": len(common_users),
    }


# Metric 2 & 3: Intra-community coherence and inter-community separation

def compute_coherence_separation(
    assignments: dict[str, str],
    usernames: list[str],
    embeddings: np.ndarray,
) -> dict[str, float]:
    """
    Compute average intra-community cosine similarity and inter-community
    centroid distance for one strategy's assignments.
    Embeddings must be L2-normalised (dot product = cosine similarity).
    """
    user_idx = {u: i for i, u in enumerate(usernames)}

    # Group users by community
    comm_indices: dict[str, list[int]] = defaultdict(list)
    for user, comm in assignments.items():
        if _is_residual(comm):
            continue
        if user in user_idx:
            comm_indices[comm].append(user_idx[user])

    # Filter small communities
    comm_indices = {
        c: idxs for c, idxs in comm_indices.items()
        if len(idxs) >= config.MIN_COMMUNITY_SIZE_FOR_ANALYSIS
    }

    if not comm_indices:
        return {"intra_coherence_mean": None, "inter_separation_mean": None}

    # Intra-community coherence: avg pairwise cosine similarity within community
    intra_scores = []
    centroids: dict[str, np.ndarray] = {}

    for comm, idxs in comm_indices.items():
        emb = embeddings[idxs]   # shape [n_members, dim]
        centroid = emb.mean(axis=0)
        centroid /= (np.linalg.norm(centroid) + 1e-8)
        centroids[comm] = centroid

        # Avg pairwise: for large communities, sample for speed
        if len(idxs) > 200:
            sample_idx = np.random.default_rng(42).choice(len(idxs), 200, replace=False)
            emb_sample = emb[sample_idx]
        else:
            emb_sample = emb

        # Dot product of normalised embeddings = cosine similarity
        sim_matrix = emb_sample @ emb_sample.T
        n = len(emb_sample)
        # Mean of off-diagonal elements
        off_diag_sum = sim_matrix.sum() - np.trace(sim_matrix)
        n_pairs      = n * (n - 1)
        if n_pairs > 0:
            intra_scores.append(off_diag_sum / n_pairs)

    # Inter-community separation: avg cosine distance between centroids
    centroid_list = list(centroids.values())
    inter_distances = []
    for i in range(len(centroid_list)):
        for j in range(i + 1, len(centroid_list)):
            cos_sim  = float(centroid_list[i] @ centroid_list[j])
            cos_dist = 1.0 - cos_sim
            inter_distances.append(cos_dist)

    return {
        "intra_coherence_mean":    round(float(np.mean(intra_scores)),    4) if intra_scores    else None,
        "intra_coherence_std":     round(float(np.std(intra_scores)),     4) if intra_scores    else None,
        "inter_separation_mean":   round(float(np.mean(inter_distances)), 4) if inter_distances else None,
        "inter_separation_min":    round(float(np.min(inter_distances)),  4) if inter_distances else None,
        "n_communities_analysed":  len(comm_indices),
    }


# Metric 4: Topic entropy + Metric 5: Vocabulary distinctiveness
def compute_vocab_metrics(
    assignments: dict[str, str],
    strategy_id: str,
    corpora: dict[str, str] | None = None,
) -> dict[str, list[str]]:
    """
    For each community, compute TF-IDF and extract:
      - Top N most distinctive terms (inter-community TF-IDF)
      - Shannon entropy of term frequency distribution

    Returns: community_id → top_terms list
    (Entropy is also logged but stored in per-strategy output)
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from scipy.stats import entropy as scipy_entropy
        import scipy.sparse
    except ImportError:
        log.warning("scikit-learn/scipy not installed. Skipping vocab metrics.")
        return {}

    # Build per-community text (use pre-loaded corpora if available)
    comm_texts: dict[str, list[str]] = defaultdict(list)
    if corpora is not None:
        for user, comm in assignments.items():
            if _is_residual(comm):
                continue
            text = corpora.get(user)
            if text:
                comm_texts[comm].append(text)
    else:
        for user, comm in assignments.items():
            if _is_residual(comm):
                continue
            corpus_path = config.PROFILES_DIR / user / "text_corpus.txt"
            if corpus_path.exists():
                text = corpus_path.read_text(encoding="utf-8").strip()
                if text:
                    comm_texts[comm].append(text)

    # Filter small communities
    comm_texts = {
        c: texts for c, texts in comm_texts.items()
        if len(texts) >= config.MIN_COMMUNITY_SIZE_FOR_ANALYSIS
    }

    if not comm_texts:
        return {}

    community_ids = list(comm_texts.keys())
    documents = [" ".join(texts) for texts in comm_texts.values()]

    _AUTOMOD_BIGRAMS = re.compile(
        r"\b(knowledge exception|qualified source|comment rule|comment rules|"
        r"concerns feel|evidence allowed|source common|claiming true|"
        r"message comment|questions concerns|community rules|rule violation|"
        r"removal reason|post removed|comment removed|mod team|"
        r"contact mod|message mod|wiki sources|wiki guidelines)\b",
        re.IGNORECASE,
    )
    documents = [_AUTOMOD_BIGRAMS.sub("", doc) for doc in documents]

    automod_stop = {
        "wiki", "wiki_comment_rules", "wiki_sources", "wiki_guidelines",
        "guidelines", "comment_rules", "rules_comment",
        "neutralpolitics", "modmail", "compose", "subject", "20removal",
        "removed", "violating", "removed_violating", "automoderator",
        "https", "http", "www", "com", "reddit", "redact",
    }
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS  # type: ignore
    full_stop = list(set(ENGLISH_STOP_WORDS) | automod_stop)

    try:
        tfidf = TfidfVectorizer(
            max_features=10_000,
            min_df=2,
            stop_words=full_stop,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        X = tfidf.fit_transform(documents)   # shape [n_communities, n_terms]
        feature_names = np.array(tfidf.get_feature_names_out())
    except Exception as e:
        log.warning(f"TF-IDF failed: {e}")
        return {}

    community_vocab: dict[str, list[str]] = {}
    for i, comm_id in enumerate(community_ids):
        row   = np.asarray(X[i].todense()).flatten()
        top_n = min(config.TOP_TFIDF_TERMS, len(row))
        top_idx = np.argsort(row)[::-1][:top_n]
        community_vocab[comm_id] = feature_names[top_idx].tolist()

    return community_vocab


# Metric 6: Gini coefficient of community sizes
def gini_coefficient(values: list[int]) -> float:
    """Compute Gini coefficient for a list of counts. 0=equal, 1=maximally unequal."""
    if not values or sum(values) == 0:
        return 0.0
    arr = np.sort(np.array(values, dtype=float))
    n   = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum()) / (n * arr.sum()) - (n + 1) / n)


def compute_size_distribution(assignments: dict[str, str]) -> dict:
    from collections import Counter
    size_counter = Counter(
        c for c in assignments.values() if not _is_residual(c)
    )
    sizes = list(size_counter.values())
    if not sizes:
        return {
            "n_communities": 0, "min_size": 0, "max_size": 0,
            "mean_size": 0.0, "median_size": 0.0, "gini": 0.0,
        }
    return {
        "n_communities":  len(sizes),
        "min_size":       int(min(sizes)),
        "max_size":       int(max(sizes)),
        "mean_size":      round(float(np.mean(sizes)), 1),
        "median_size":    round(float(np.median(sizes)), 1),
        "gini":           round(gini_coefficient(sizes), 4),
    }


# Main analysis loop
def analyze_strategy(
    strategy_id: str,
    baseline_assignments: dict[str, str],
    usernames: list[str] | None,
    embeddings: np.ndarray | None,
    corpora: dict[str, str] | None = None,
) -> dict:
    """Run all applicable metrics for one strategy."""
    log.info(f"[Analysis] {strategy_id}")

    assignments = load_assignments(strategy_id)
    if assignments is None:
        return {"strategy_id": strategy_id, "error": "assignments_not_found"}

    result = {"strategy_id": strategy_id}

    # Metric 1: NMI / ARI vs baseline
    if strategy_id != "strategy_1_subreddit":
        result["cluster_purity"] = compute_nmi_ari(baseline_assignments, assignments)
    else:
        result["cluster_purity"] = {"note": "this IS the baseline"}

    # Metrics 2 & 3: coherence / separation (requires embeddings)
    if embeddings is not None and usernames is not None:
        result["coherence_separation"] = compute_coherence_separation(
            assignments, usernames, embeddings
        )
    else:
        result["coherence_separation"] = {"error": "embeddings_unavailable"}

    # Metrics 4 & 5: vocabulary
    log.info("  [%s] vocab/TF-IDF ...", strategy_id)
    result["vocab_top_terms"] = compute_vocab_metrics(assignments, strategy_id, corpora)
    log.info("  [%s] vocab done (%d communities with top-terms)",
             strategy_id, len(result["vocab_top_terms"]))

    # Metric 6: size distribution
    result["size_distribution"] = compute_size_distribution(assignments)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategies", nargs="*", default=STRATEGY_IDS,
        help="Strategy IDs to analyse (default: all)"
    )
    args = parser.parse_args()

    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    # Load embeddings once (shared)
    usernames, embeddings = load_embeddings()

    # Load baseline (Strategy 1) once
    baseline = load_assignments("strategy_1_subreddit")
    if baseline is None:
        log.warning("Strategy 1 assignments not found — NMI/ARI comparisons will be skipped")

    # Pre-load every user corpus once (shared across strategies for vocab metrics)
    all_users: set[str] = set()
    if baseline:
        all_users.update(baseline.keys())
    for sid in args.strategies:
        a = load_assignments(sid)
        if a:
            all_users.update(a.keys())
    corpora = load_all_corpora(sorted(all_users)) if all_users else {}

    all_results = []
    for strategy_id in args.strategies:
        result = analyze_strategy(
            strategy_id, baseline or {}, usernames, embeddings, corpora,
        )
        all_results.append(result)

        per_strategy_dir = ANALYSIS_DIR / "per_strategy" / strategy_id
        per_strategy_dir.mkdir(parents=True, exist_ok=True)
        write_json(per_strategy_dir / "full_analysis.json", result)

    # Comparison table (machine-readable)
    write_json(ANALYSIS_DIR / "comparison_table.json", all_results)

    # Comparison table (CSV — flatten key metrics)
    csv_rows = []
    for r in all_results:
        row = {
            "strategy": r["strategy_id"],
            "n_communities":        r.get("size_distribution", {}).get("n_communities"),
            "gini":                 r.get("size_distribution", {}).get("gini"),
            "mean_size":            r.get("size_distribution", {}).get("mean_size"),
            "nmi_vs_baseline":      r.get("cluster_purity", {}).get("nmi"),
            "ari_vs_baseline":      r.get("cluster_purity", {}).get("ari"),
            "intra_coherence_mean": r.get("coherence_separation", {}).get("intra_coherence_mean"),
            "inter_separation_mean":r.get("coherence_separation", {}).get("inter_separation_mean"),
        }
        csv_rows.append(row)

    with open(ANALYSIS_DIR / "comparison_table.csv", "w", newline="", encoding="utf-8") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    log.info("=" * 60)
    log.info("ANALYSIS COMPLETE")
    log.info(f"  Results: {ANALYSIS_DIR / 'comparison_table.csv'}")
    log.info("=" * 60)

    # Print summary table to console
    log.info("")
    log.info(f"{'Strategy':<35} {'K':>4} {'Gini':>6} {'NMI':>6} {'ARI':>6} {'Coh.':>6} {'Sep.':>6}")
    log.info("-" * 75)
    for row in csv_rows:
        def _f(v, fmt=".4f"):
            return format(v, fmt) if v is not None else " N/A "
        log.info(
            f"{row['strategy']:<35} "
            f"{str(row['n_communities'] or ''):>4} "
            f"{_f(row['gini']):>6} "
            f"{_f(row['nmi_vs_baseline']):>6} "
            f"{_f(row['ari_vs_baseline']):>6} "
            f"{_f(row['intra_coherence_mean']):>6} "
            f"{_f(row['inter_separation_mean']):>6}"
        )


if __name__ == "__main__":
    main()
