"""
Strategy 4: Hybrid grouping (graph + semantic, sparse + Leiden).

    S_hybrid[i, j] = Alpha · S_graph[i, j]  +  (1 − Alpha) · S_semantic[i, j]

For tractability at n > 10 000 users, S_hybrid is materialised only on the
nnz pattern of S_graph (the sparse user-user projection produced by the same
chunked accumulator as Strategy 2). S_semantic[i, j] is computed lazily on
those entries as cosine similarity of the L2-normalised user embeddings
reused from Strategy 3 — never as a dense n × n matrix.

Communities are then found by Leiden on the resulting weighted sparse
graph, run once per Alpha in config.HYBRID_ALPHA_VALUES.
"""

import logging
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

STRATEGY_ID = "strategy_4_hybrid"
OUTPUT_DIR  = config.DATA_DIR / "groupings" / STRATEGY_ID

# Chunk size when computing cosine similarities on sparse-graph edges.
COSINE_CHUNK_EDGES: int = 5_000_000


def _load_embeddings():
    emb_path   = config.DATA_DIR / "groupings" / "strategy_3_semantic" / "user_embeddings.npy"
    order_path = config.DATA_DIR / "groupings" / "strategy_3_semantic" / "user_order.json"
    if emb_path.exists() and order_path.exists():
        log.info("[Strategy 4] Reusing Strategy 3 embeddings")
        return read_json(order_path), np.load(str(emb_path))
    return None, None


def _build_sparse_graph_for_users(
    usernames: list[str],
    max_users_per_sub: int,
    min_edge_weight: float,
    flush_every: int,
):
    """
    Build the sparse user-user projection for `usernames` (preserving the
    given index order). Returns (sub_inter_csr, sub_union_csr) as scipy
    sparse CSR matrices of intersection-sums and union-sums; pointwise
    Jaccard = inter / union on the shared nnz pattern.
    """
    import scipy.sparse as sp

    # Build per-sub user lists in the *given* index order
    user_to_idx = {u: i for i, u in enumerate(usernames)}
    n = len(usernames)

    sub_to_users: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for u in usernames:
        ap = config.PROFILES_DIR / u / "activity.json"
        if not ap.exists():
            continue
        i = user_to_idx[u]
        for sub, rec in read_json(ap).items():
            w = rec.get("posts", 0) + rec.get("comments", 0)
            if w > 0:
                sub_to_users[sub].append((i, float(w)))

    log.info(f"[Strategy 4] {len(sub_to_users):,} subs, projecting (cap={max_users_per_sub})")

    inter = sp.csr_matrix((n, n), dtype=np.float32)
    union = sp.csr_matrix((n, n), dtype=np.float32)
    rows: list[int] = []
    cols: list[int] = []
    inter_d: list[float] = []
    union_d: list[float] = []
    pending = 0

    def _flush():
        nonlocal inter, union, rows, cols, inter_d, union_d, pending
        if not rows:
            return
        r = np.asarray(rows, dtype=np.int32)
        c = np.asarray(cols, dtype=np.int32)
        coo_i = sp.coo_matrix((np.asarray(inter_d, dtype=np.float32), (r, c)), shape=(n, n))
        coo_u = sp.coo_matrix((np.asarray(union_d, dtype=np.float32), (r, c)), shape=(n, n))
        coo_i.sum_duplicates(); coo_u.sum_duplicates()
        inter = (inter + coo_i.tocsr()).tocsr()
        union = (union + coo_u.tocsr()).tocsr()
        rows.clear(); cols.clear(); inter_d.clear(); union_d.clear()
        pending = 0

    sub_items = sorted(sub_to_users.items(), key=lambda kv: -len(kv[1]))
    total_emit = 0
    for sub_idx, (sub, plist) in enumerate(sub_items):
        if len(plist) < 2:
            continue
        if len(plist) > max_users_per_sub:
            plist = sorted(plist, key=lambda x: -x[1])[:max_users_per_sub]
        n_kept = len(plist)
        idxs = np.fromiter((p[0] for p in plist), dtype=np.int32, count=n_kept)
        wts  = np.fromiter((p[1] for p in plist), dtype=np.float32, count=n_kept)

        for a in range(n_kept - 1):
            ia = idxs[a]; wa = wts[a]
            jb = idxs[a + 1:]; wb = wts[a + 1:]
            mn = np.minimum(wa, wb)
            mx = np.maximum(wa, wb)
            ii = np.minimum(ia, jb)
            jj = np.maximum(ia, jb)
            rows.extend(ii.tolist())
            cols.extend(jj.tolist())
            inter_d.extend(mn.tolist())
            union_d.extend(mx.tolist())
            pending += jb.shape[0]
            total_emit += jb.shape[0]
            if pending >= flush_every:
                log.info(
                    f"[Strategy 4]   flush — pending={pending:,} "
                    f"(emitted={total_emit:,}, inter nnz={inter.nnz:,})"
                )
                _flush()

        if (sub_idx + 1) % 20 == 0 or sub_idx == len(sub_items) - 1:
            log.info(
                f"[Strategy 4]   {sub_idx + 1}/{len(sub_items)} subs, "
                f"emitted={total_emit:,}, inter nnz={inter.nnz:,}"
            )
    _flush()

    # Symmetrise (we stored upper-triangle only)
    inter = inter + inter.T
    union = union + union.T
    log.info(f"[Strategy 4] Symmetric inter/union nnz={inter.nnz:,}")

    # Threshold by intersection weight (keeps at least min_edge_weight summed-min)
    if min_edge_weight > 0:
        before = inter.nnz
        mask = inter.data >= min_edge_weight
        inter.data = inter.data * mask
        inter.eliminate_zeros()
        # Apply same mask to union by intersecting nnz patterns
        union = union.multiply(inter > 0).tocsr()
        log.info(f"[Strategy 4] Threshold inter >= {min_edge_weight}: {before:,} → {inter.nnz:,} nnz")

    return inter, union


def _cosine_on_edges(emb: np.ndarray, csr) -> np.ndarray:
    """
    Compute cosine similarity emb[i] · emb[j] for every (i,j) in the CSR
    nnz pattern. Returns a 1-D array aligned with csr.tocoo().data order.
    Uses chunked dot products to bound memory.
    """
    coo = csr.tocoo()
    rows = coo.row
    cols = coo.col
    n_edges = rows.shape[0]
    out = np.empty(n_edges, dtype=np.float32)
    for start in range(0, n_edges, COSINE_CHUNK_EDGES):
        end = min(start + COSINE_CHUNK_EDGES, n_edges)
        a = emb[rows[start:end]]
        b = emb[cols[start:end]]
        out[start:end] = (a * b).sum(axis=1)
    np.clip(out, 0.0, 1.0, out=out)
    return out


def _leiden_partition(csr, n_users: int, weight_col: str = "weight") -> tuple[dict[int, int], float]:
    try:
        import igraph as ig
        import leidenalg as la
        import scipy.sparse as sp
        coo = sp.triu(csr, k=1).tocoo()
        g = ig.Graph(n=n_users, edges=list(zip(coo.row.tolist(), coo.col.tolist())), directed=False)
        g.es[weight_col] = coo.data.tolist()
        log.info(f"[Strategy 4]   G: |V|={n_users:,}, |E|={coo.nnz:,}")
        part = la.find_partition(
            g, la.RBConfigurationVertexPartition,
            weights=weight_col, resolution_parameter=config.GRAPH_RESOLUTION, seed=42,
        )
        return {i: part.membership[i] for i in range(n_users)}, float(part.modularity)
    except ImportError:
        import networkx as nx
        import community as community_louvain
        G = nx.from_scipy_sparse_array(csr, edge_attribute=weight_col)
        G.add_nodes_from(range(n_users))
        log.info(f"[Strategy 4]   G: |V|={G.number_of_nodes():,}, |E|={G.number_of_edges():,}")
        part = community_louvain.best_partition(
            G, weight=weight_col, resolution=config.GRAPH_RESOLUTION, random_state=42,
        )
        return part, community_louvain.modularity(part, G, weight=weight_col)


def _run_for_alpha(
    usernames: list[str],
    inter,            # scipy.sparse.csr_matrix (intersection sums)
    union,            # scipy.sparse.csr_matrix (union sums, same nnz pattern)
    sem_data: np.ndarray,  # cosine sims aligned with inter.tocoo().data order
    alpha: float,
    out_dir: Path,
) -> dict:
    import scipy.sparse as sp

    log.info(f"[Strategy 4] α={alpha:.2f}: combining sparse similarities")
    coo = inter.tocoo()
    # Jaccard in [0, 1] on the nnz pattern
    union_data = union.tocoo().data  # same shape as coo.data
    safe_union = np.where(union_data > 0, union_data, 1.0)
    graph_sim = coo.data / safe_union
    np.clip(graph_sim, 0.0, 1.0, out=graph_sim)

    hybrid_data = (alpha * graph_sim + (1.0 - alpha) * sem_data).astype(np.float32)
    n = inter.shape[0]
    H = sp.coo_matrix((hybrid_data, (coo.row, coo.col)), shape=(n, n)).tocsr()

    part_idx, mod = _leiden_partition(H, n_users=n)
    partition: dict[str, int | str] = {usernames[i]: c for i, c in part_idx.items()}
    n_raw_comms = len(set(partition.values()))

    # Consolidate to top-K + "other" for cross-strategy comparability
    if config.TOP_K_COMMUNITIES > 0:
        from grouping._consolidate import consolidate_top_k
        partition, n_kept, n_other_members = consolidate_top_k(
            partition, top_k=config.TOP_K_COMMUNITIES, other_label="other",
        )
        log.info(
            "[Strategy 4] α=%.2f: consolidated %d → %d communities (%d → 'other')",
            alpha, n_raw_comms, n_kept + (1 if n_other_members > 0 else 0),
            n_other_members,
        )

    community_members: dict[int | str, list[str]] = defaultdict(list)
    for u, c in partition.items():
        community_members[c].append(u)

    community_profiles = []
    for cid, members in sorted(community_members.items(), key=lambda x: -len(x[1])):
        community_profiles.append({
            "community_id": f"hybrid_{cid}",
            "n_members":    len(members),
            "members":      members,
        })

    assignments = {u: [f"hybrid_{c}"] for u, c in partition.items()}

    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(out_dir / "community_assignments.json", assignments)
    write_json(out_dir / "community_profiles.json",   community_profiles)
    meta = {
        "strategy":          STRATEGY_ID,
        "alpha":             alpha,
        "top_k_communities": config.TOP_K_COMMUNITIES,
        "n_users":           n,
        "n_raw_communities": n_raw_comms,
        "n_communities":     len(community_members),
        "modularity":        mod,
        "run_at":            datetime.now(timezone.utc).isoformat(),
    }
    write_json(out_dir / "grouping_meta.json", meta)
    return meta


def run(user_index: list[dict]) -> dict:
    """
    Hybrid community detection over the full alpha sweep, sparse-throughout.
    """
    usernames, embeddings = _load_embeddings()
    if embeddings is None:
        log.info("[Strategy 4] No cached embeddings — encoding inline")
        from grouping.strategy_semantic import _load_user_texts, _encode_texts
        usernames, texts = _load_user_texts(user_index)
        embeddings = _encode_texts(texts, config.EMBEDDING_MODEL)
    else:
        # Restrict user_index to users with embeddings (Strategy 3 already filtered)
        emb_users = set(usernames)
        user_index = [p for p in user_index if p["username"] in emb_users]

    log.info(f"[Strategy 4] Building sparse graph similarity ({len(usernames):,} users)")
    inter, union = _build_sparse_graph_for_users(
        usernames,
        max_users_per_sub=config.GRAPH_MAX_USERS_PER_SUB,
        min_edge_weight=config.GRAPH_MIN_EDGE_WEIGHT,
        flush_every=config.GRAPH_FLUSH_EVERY_PAIRS,
    )
    log.info(f"[Strategy 4] Sparse graph built — nnz={inter.nnz:,}")

    log.info("[Strategy 4] Computing cosine similarity on sparse-graph edges")
    sem_data = _cosine_on_edges(embeddings, inter)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    all_metas = []
    for alpha in config.HYBRID_ALPHA_VALUES:
        alpha_dir = OUTPUT_DIR / "alpha_sweep" / f"alpha_{alpha}"
        meta = _run_for_alpha(usernames, inter, union, sem_data, alpha, alpha_dir)
        all_metas.append(meta)
        log.info(
            f"[Strategy 4] α={alpha:.2f} done — "
            f"{meta['n_communities']:,} communities, modularity={meta['modularity']:.4f}"
        )

    # Mirror the default-alpha run at OUTPUT_DIR root for convenience
    default_alpha_dir = OUTPUT_DIR / "alpha_sweep" / f"alpha_{config.HYBRID_ALPHA_DEFAULT}"
    if default_alpha_dir.exists():
        import shutil
        for fname in ["community_assignments.json", "community_profiles.json", "grouping_meta.json"]:
            src = default_alpha_dir / fname
            if src.exists():
                shutil.copy(src, OUTPUT_DIR / fname)

    write_json(OUTPUT_DIR / "alpha_sweep_summary.json", all_metas)

    return {
        "meta": {
            "strategy":         STRATEGY_ID,
            "alpha_values_run": config.HYBRID_ALPHA_VALUES,
            "default_alpha":    config.HYBRID_ALPHA_DEFAULT,
            "n_users":          len(usernames),
            "run_at":           datetime.now(timezone.utc).isoformat(),
        }
    }


def main():
    user_index_path = config.DATA_DIR / "user_index.json"
    if not user_index_path.exists():
        log.error("user_index.json not found. Run `python run.py profiles` first.")
        return
    user_index = read_json(user_index_path)
    run(user_index)
    log.info(f"[Strategy 4] Done — alpha sweep complete. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
