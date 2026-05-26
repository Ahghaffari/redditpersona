"""
Strategy 5: User-Interaction Graph Grouping (Leiden)
=====================================================
Builds a directed user->user reply graph from data/user_interactions.jsonl
(emitted in Step 1 by ActivityTracker.record_reply). Edge weight is the
number of times user A replied to user B, summed over all subreddits.

We run Leiden community detection (Traag et al. 2019), the modern successor
of Louvain. Falls back to Louvain if leidenalg is unavailable.

Output:
  data/groupings/strategy_5_interaction/
    community_assignments.json
    community_profiles.json
    grouping_meta.json
    user_graph.graphml
"""

import logging
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from utils import ensure_dirs, write_json, read_json, iter_jsonl

ensure_dirs()
logging.basicConfig(level=config.LOG_LEVEL, format="%(asctime)s  %(levelname)-8s  %(message)s")
log = logging.getLogger(__name__)

STRATEGY_ID = "strategy_5_interaction"
OUTPUT_DIR  = config.DATA_DIR / "groupings" / STRATEGY_ID


def _load_reply_edges(allowed_users: set[str]) -> dict[tuple[str, str], int]:
    """Stream user_interactions.jsonl into an undirected weighted edge dict."""
    edges: dict[tuple[str, str], int] = defaultdict(int)
    if not config.INTERACTION_FILE.exists():
        log.error(f"{config.INTERACTION_FILE} not found. Run Step 1 first.")
        return edges

    n = 0
    for rec in iter_jsonl(config.INTERACTION_FILE):
        u, v = rec.get("from"), rec.get("to")
        if not u or not v or u == v:
            continue
        if u not in allowed_users or v not in allowed_users:
            continue
        w = int(rec.get("weight", 1))
        if w < config.INTERACTION_MIN_EDGE_WEIGHT:
            continue
        key = (u, v) if u < v else (v, u)   # undirected aggregation
        edges[key] += w
        n += 1
        if n % 1_000_000 == 0:
            log.info(f"[Strategy 5] {n:,} reply edges streamed, {len(edges):,} unique pairs")
    log.info(f"[Strategy 5] {n:,} total replies → {len(edges):,} unique user pairs")
    return edges


def _detect_communities(G):
    """Leiden if available (igraph + leidenalg); else python-louvain."""
    try:
        import igraph as ig
        import leidenalg as la
        log.info("[Strategy 5] Leiden community detection")
        nodes = list(G.nodes())
        idx = {n: i for i, n in enumerate(nodes)}
        edges = [(idx[u], idx[v]) for u, v in G.edges()]
        weights = [G[u][v]["weight"] for u, v in G.edges()]
        g = ig.Graph(n=len(nodes), edges=edges, directed=False)
        g.es["weight"] = weights
        part = la.find_partition(
            g, la.RBConfigurationVertexPartition,
            weights="weight", resolution_parameter=config.LEIDEN_RESOLUTION,
            seed=42,
        )
        partition = {nodes[i]: part.membership[i] for i in range(len(nodes))}
        modularity = float(part.modularity)
        algorithm = "Leiden"
    except ImportError:
        import community as community_louvain
        log.info("[Strategy 5] leidenalg unavailable — falling back to Louvain")
        partition = community_louvain.best_partition(
            G, weight="weight", resolution=config.LEIDEN_RESOLUTION, random_state=42,
        )
        modularity = community_louvain.modularity(partition, G, weight="weight")
        algorithm = "Louvain"
    return partition, modularity, algorithm


def run(user_index: list[dict]) -> dict:
    import networkx as nx

    allowed = {p["username"] for p in user_index}
    edges = _load_reply_edges(allowed)
    if not edges:
        return {
            "assignments": {}, "communities": [],
            "meta": {"strategy": STRATEGY_ID, "n_users": 0, "n_communities": 0},
        }

    G = nx.Graph()
    G.add_nodes_from(allowed)
    for (u, v), w in edges.items():
        G.add_edge(u, v, weight=float(w))

    partition, modularity, algorithm = _detect_communities(G)
    n_raw_comms = len(set(partition.values()))

    # Consolidate to top-K + "other" so K is comparable across strategies
    if config.TOP_K_COMMUNITIES > 0:
        from grouping._consolidate import consolidate_top_k
        partition, n_kept, n_other_members = consolidate_top_k(
            partition, top_k=config.TOP_K_COMMUNITIES, other_label="other",
        )
        log.info(
            "[Strategy 5] Consolidated %d → %d communities (%d members → 'other')",
            n_raw_comms, n_kept + (1 if n_other_members > 0 else 0), n_other_members,
        )

    members: dict[int | str, list[str]] = defaultdict(list)
    for user, cid in partition.items():
        members[cid].append(user)

    community_profiles = []
    for cid, mlist in sorted(members.items(), key=lambda x: -len(x[1])):
        sub_totals: dict[str, float] = defaultdict(float)
        for user in mlist:
            ap = config.PROFILES_DIR / user / "activity.json"
            if ap.exists():
                for sub, rec in read_json(ap).items():
                    sub_totals[sub] += rec.get("posts", 0) + rec.get("comments", 0)
        top_subs = sorted(sub_totals.items(), key=lambda x: -x[1])[:10]
        cat_votes: dict[str, float] = defaultdict(float)
        for sub, w in sub_totals.items():
            cat_votes[config.SUB_TO_CATEGORY.get(sub, "unknown")] += w
        dominant = max(cat_votes, key=cat_votes.get) if cat_votes else "unknown"
        community_profiles.append({
            "community_id":      f"intx_{cid}",
            "n_members":         len(mlist),
            "members":           mlist,
            "dominant_category": dominant,
            "top_subreddits":    [s for s, _ in top_subs],
        })

    assignments = {u: [f"intx_{c}"] for u, c in partition.items()}

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        nx.write_graphml(G, str(OUTPUT_DIR / "user_graph.graphml"))
    except Exception as e:
        log.warning(f"Could not save GraphML: {e}")

    return {
        "assignments": assignments,
        "communities": community_profiles,
        "meta": {
            "strategy":          STRATEGY_ID,
            "algorithm":         algorithm,
            "leiden_resolution": config.LEIDEN_RESOLUTION,
            "top_k_communities": config.TOP_K_COMMUNITIES,
            "n_users":           len(partition),
            "n_raw_communities": n_raw_comms,
            "n_communities":     len(members),
            "n_edges":           G.number_of_edges(),
            "modularity":        modularity,
            "run_at":            datetime.now(timezone.utc).isoformat(),
        },
    }


def main():
    user_index_path = config.DATA_DIR / "user_index.json"
    if not user_index_path.exists():
        log.error("user_index.json not found. Run `python run.py profiles` first.")
        return
    user_index = read_json(user_index_path)
    log.info(f"[Strategy 5] Running on {len(user_index)} users")

    result = run(user_index)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json(OUTPUT_DIR / "community_assignments.json", result["assignments"])
    write_json(OUTPUT_DIR / "community_profiles.json",   result["communities"])
    write_json(OUTPUT_DIR / "grouping_meta.json",        result["meta"])
    log.info(
        f"[Strategy 5] Done — {result['meta']['n_communities']} communities "
        f"(modularity={result['meta'].get('modularity', 0):.4f})"
    )


if __name__ == "__main__":
    main()
