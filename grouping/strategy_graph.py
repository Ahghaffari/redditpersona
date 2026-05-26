"""
Strategy 2: Graph-structural grouping (Leiden on bipartite projection).

Builds a weighted user <-> subreddit bipartite graph (weight = posts+comments),
projects to a user-user graph (shared-subreddit overlap, summed min-weight),
and runs Leiden community detection at the configured resolution.

Memory-safe projection (chunked sparse accumulator):
  • For each subreddit, cap the participant set to the top
    GRAPH_MAX_USERS_PER_SUB users by activity (drops long-tail one-offs in
    hub subs like r/AskReddit, which otherwise produce C(N,2) ≈ 10⁹ pairs).
  • Append (i, j, min(w_i, w_j)) into COO arrays; every
    GRAPH_FLUSH_EVERY_PAIRS appends, sum_duplicates() and merge into a
    running scipy.sparse.csr_matrix accumulator.
  • Threshold edges below GRAPH_MIN_EDGE_WEIGHT after projection.
  • Build NetworkX graph from the sparse matrix in one pass.
"""

import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import ensure_dirs, write_json, read_json, read_jsonl

ensure_dirs()
logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

STRATEGY_ID = "strategy_2_graph"
OUTPUT_DIR  = config.DATA_DIR / "groupings" / STRATEGY_ID


def _load_sub_user_weights(user_index: list[dict]) -> tuple[list[str], dict[str, list[tuple[int, float]]]]:
    """
    Single pass over activity.json files.

    Returns
    -------
    usernames     : list of usernames (index = node id)
    sub_to_users  : sub → list of (user_idx, weight), unsorted
    """
    usernames: list[str] = []
    sub_to_users: dict[str, list[tuple[int, float]]] = defaultdict(list)

    for i, profile in enumerate(user_index):
        username = profile["username"]
        usernames.append(username)
        ap = config.PROFILES_DIR / username / "activity.json"
        if not ap.exists():
            continue
        for sub, rec in read_json(ap).items():
            w = rec.get("posts", 0) + rec.get("comments", 0)
            if w > 0:
                sub_to_users[sub].append((i, float(w)))

    return usernames, sub_to_users


def _project_to_user_graph_sparse(
    n_users: int,
    sub_to_users: dict[str, list[tuple[int, float]]],
    max_users_per_sub: int,
    min_edge_weight: float,
    flush_every: int,
):
    """
    Memory-safe bipartite → user-user projection.

    Returns scipy.sparse.csr_matrix (symmetric, n_users × n_users), float32,
    where M[i, j] = Σ_sub min(w_{i,sub}, w_{j,sub}) over subs both users
    appear in (after the per-sub user cap).
    """
    import scipy.sparse as sp

    accum: sp.csr_matrix = sp.csr_matrix((n_users, n_users), dtype=np.float32)
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    pending = 0
    total_pairs_emitted = 0

    def _flush():
        nonlocal accum, rows, cols, data, pending
        if not rows:
            return
        coo = sp.coo_matrix(
            (np.asarray(data, dtype=np.float32),
             (np.asarray(rows, dtype=np.int32),
              np.asarray(cols, dtype=np.int32))),
            shape=(n_users, n_users),
        )
        coo.sum_duplicates()
        accum = (accum + coo.tocsr()).tocsr()
        rows.clear(); cols.clear(); data.clear()
        pending = 0

    sub_items = sorted(sub_to_users.items(), key=lambda kv: -len(kv[1]))
    for sub_idx, (sub, plist) in enumerate(sub_items):
        n_in_sub = len(plist)
        if n_in_sub < 2:
            continue
        # Cap to top-K most-active users in this sub
        if n_in_sub > max_users_per_sub:
            plist = sorted(plist, key=lambda x: -x[1])[:max_users_per_sub]
            n_kept = max_users_per_sub
            log.info(
                f"[Strategy 2]   sub '{sub}': {n_in_sub:,} users → capped to "
                f"top {n_kept:,} (will emit {n_kept*(n_kept-1)//2:,} pairs)"
            )
        else:
            n_kept = n_in_sub

        # Pre-extract idx & weight arrays for speed
        idxs = np.fromiter((p[0] for p in plist), dtype=np.int32, count=n_kept)
        wts  = np.fromiter((p[1] for p in plist), dtype=np.float32, count=n_kept)

        # All-pairs (upper-triangular): use a tight loop with numpy minimum
        for a in range(n_kept - 1):
            ia = idxs[a]; wa = wts[a]
            jb = idxs[a + 1:]
            wb = wts[a + 1:]
            mn = np.minimum(wa, wb)
            # Upper-triangle order by node id
            ii = np.minimum(ia, jb)
            jj = np.maximum(ia, jb)
            rows.extend(ii.tolist())
            cols.extend(jj.tolist())
            data.extend(mn.tolist())
            pending += jb.shape[0]
            total_pairs_emitted += jb.shape[0]

            if pending >= flush_every:
                log.info(
                    f"[Strategy 2]   flush — pending={pending:,} "
                    f"(total emitted={total_pairs_emitted:,}, accum nnz={accum.nnz:,})"
                )
                _flush()

        if (sub_idx + 1) % 10 == 0 or sub_idx == len(sub_items) - 1:
            log.info(
                f"[Strategy 2]   processed {sub_idx + 1}/{len(sub_items)} subs, "
                f"emitted {total_pairs_emitted:,} pairs, accum nnz={accum.nnz:,}"
            )

    _flush()

    # Symmetrise (we stored only upper-triangle entries)
    log.info(f"[Strategy 2] Symmetrising — upper nnz={accum.nnz:,}")
    accum = accum + accum.T
    log.info(f"[Strategy 2] Full nnz={accum.nnz:,}")

    # Threshold weak edges
    if min_edge_weight > 0:
        before = accum.nnz
        accum.data[accum.data < min_edge_weight] = 0.0
        accum.eliminate_zeros()
        log.info(
            f"[Strategy 2] Threshold w >= {min_edge_weight}: "
            f"{before:,} → {accum.nnz:,} nnz"
        )

    return accum


def run(user_index: list[dict]) -> dict:
    """
    Execute graph-structural community detection.

    Requires: igraph, leidenalg, scipy (falls back to networkx + python-louvain).
    """

    log.info(
        f"[Strategy 2] Loading per-sub user weights from {len(user_index):,} profiles..."
    )
    usernames, sub_to_users = _load_sub_user_weights(user_index)
    n_users = len(usernames)
    n_bipartite_edges = sum(len(v) for v in sub_to_users.values())
    log.info(
        f"[Strategy 2] {n_bipartite_edges:,} bipartite edges across "
        f"{len(sub_to_users):,} subs"
    )

    log.info(
        f"[Strategy 2] Projecting (max_users_per_sub={config.GRAPH_MAX_USERS_PER_SUB:,}, "
        f"min_edge_weight={config.GRAPH_MIN_EDGE_WEIGHT}, "
        f"flush_every={config.GRAPH_FLUSH_EVERY_PAIRS:,})"
    )
    M = _project_to_user_graph_sparse(
        n_users,
        sub_to_users,
        max_users_per_sub=config.GRAPH_MAX_USERS_PER_SUB,
        min_edge_weight=config.GRAPH_MIN_EDGE_WEIGHT,
        flush_every=config.GRAPH_FLUSH_EVERY_PAIRS,
    )
    n_user_edges = M.nnz // 2
    log.info(f"[Strategy 2] {n_user_edges:,} user-user edges (after threshold)")
    log.info(f"[Strategy 2] |V|={n_users:,}, |E|={n_user_edges:,}")

    log.info(f"[Strategy 2] Running Leiden (resolution={config.GRAPH_RESOLUTION})...")
    try:
        import igraph as ig
        import leidenalg as la
        import scipy.sparse as sp
        coo = sp.triu(M, k=1).tocoo()
        g = ig.Graph(n=n_users, edges=list(zip(coo.row.tolist(), coo.col.tolist())), directed=False)
        g.es["weight"] = coo.data.tolist()
        del M, coo
        part = la.find_partition(
            g, la.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=config.GRAPH_RESOLUTION, seed=42,
        )
        partition_idx: dict[int, int] = {i: part.membership[i] for i in range(n_users)}
        modularity = float(part.modularity)
        algorithm = "Leiden"
    except ImportError:
        import networkx as nx
        import community as community_louvain
        G = nx.from_scipy_sparse_array(M, edge_attribute="weight")
        G.add_nodes_from(range(n_users))
        del M
        partition_idx = community_louvain.best_partition(
            G, weight="weight", resolution=config.GRAPH_RESOLUTION, random_state=42,
        )
        modularity = community_louvain.modularity(partition_idx, G, weight="weight")
        algorithm = "Louvain"

    # Map node-idx → username
    partition: dict[str, int | str] = {usernames[i]: c for i, c in partition_idx.items()}
    n_raw_comms = len(set(partition.values()))

    # Consolidate to top-K + "other" so K is comparable across strategies
    if config.TOP_K_COMMUNITIES > 0:
        from grouping._consolidate import consolidate_top_k
        partition, n_kept, n_other_members = consolidate_top_k(
            partition, top_k=config.TOP_K_COMMUNITIES, other_label="other",
        )
        log.info(
            "[Strategy 2] Consolidated %d → %d communities (%d members → 'other')",
            n_raw_comms, n_kept + (1 if n_other_members > 0 else 0), n_other_members,
        )

    community_members: dict[int | str, list[str]] = defaultdict(list)
    for user, comm_id in partition.items():
        community_members[comm_id].append(user)

    # Build community profiles (dominant subreddit / category per community)
    community_profiles = []
    for comm_id, members in sorted(community_members.items(), key=lambda x: -len(x[1])):
        sub_totals: dict[str, float] = defaultdict(float)
        for user in members:
            activity_path = config.PROFILES_DIR / user / "activity.json"
            if activity_path.exists():
                for sub, rec in read_json(activity_path).items():
                    sub_totals[sub] += rec.get("posts", 0) + rec.get("comments", 0)

        top_subs = sorted(sub_totals.items(), key=lambda x: -x[1])[:10]
        category_votes: dict[str, float] = defaultdict(float)
        for sub, w in sub_totals.items():
            cat = config.SUB_TO_CATEGORY.get(sub, "unknown")
            category_votes[cat] += w
        dominant_category = max(category_votes, key=category_votes.get) if category_votes else "unknown"

        community_profiles.append({
            "community_id":       f"graph_{comm_id}",
            "n_members":          len(members),
            "members":            members,
            "dominant_category":  dominant_category,
            "top_subreddits":     [s for s, _ in top_subs],
        })

    assignments = {user: [f"graph_{cid}"] for user, cid in partition.items()}

    return {
        "assignments": assignments,
        "communities": community_profiles,
        "meta": {
            "strategy":              STRATEGY_ID,
            "algorithm":             algorithm,
            "graph_resolution":      config.GRAPH_RESOLUTION,
            "max_users_per_sub":     config.GRAPH_MAX_USERS_PER_SUB,
            "min_edge_weight":       config.GRAPH_MIN_EDGE_WEIGHT,
            "top_k_communities":     config.TOP_K_COMMUNITIES,
            "n_users":               len(partition),
            "n_user_edges":          n_user_edges,
            "n_raw_communities":     n_raw_comms,
            "n_communities":         len(community_members),
            "modularity":            modularity,
            "run_at":                datetime.now(timezone.utc).isoformat(),
        },
    }


def main():
    user_index_path = config.DATA_DIR / "user_index.json"
    if not user_index_path.exists():
        log.error("user_index.json not found. Run `python run.py profiles` first.")
        return

    user_index = read_json(user_index_path)
    log.info(f"[Strategy 2] Running on {len(user_index)} users")

    result = run(user_index)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "community_assignments.json", result["assignments"])
    write_json(OUTPUT_DIR / "community_profiles.json",   result["communities"])
    write_json(OUTPUT_DIR / "grouping_meta.json",        result["meta"])

    log.info(
        f"[Strategy 2] Done — {result['meta']['n_communities']} communities "
        f"(modularity={result['meta'].get('modularity', 'N/A'):.4f})"
    )


if __name__ == "__main__":
    main()
