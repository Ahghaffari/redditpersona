"""
Post-process community partitions to a target K.

Many community-detection algorithms (Louvain, Leiden) on sparse user graphs
naturally produce a long tail of singleton/tiny communities. For the paper
we want K in a comparable range (50–150) across strategies, so we
consolidate: keep the top-K largest communities verbatim, and merge every
member of a smaller community into a single "<prefix>_other" bucket.

exactly min(top_k, n_original) + 1 communities (the +1 only added
when at least one community is merged out).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Hashable
from typing import TypeVar

T = TypeVar("T", bound=Hashable)
C = TypeVar("C", bound=Hashable)


def consolidate_top_k(
    partition: dict[T, C],
    top_k: int,
    other_label: C,
) -> tuple[dict[T, C], int, int]:
    """Keep top-K communities by member count; merge the rest under `other_label`.

    Parameters
    ----------
    partition    : member → community_id
    top_k        : number of largest communities to preserve as-is
    other_label  : label assigned to every member whose community was dropped

    Returns
    -------
    (new_partition, n_kept, n_merged_in_other)
        n_kept           — number of preserved community ids (≤ top_k)
        n_merged_in_other— number of members re-labelled to `other_label`
    """
    if top_k <= 0:
        return dict(partition), 0, 0

    sizes = Counter(partition.values())
    if len(sizes) <= top_k:
        return dict(partition), len(sizes), 0

    keep = {cid for cid, _ in sizes.most_common(top_k)}
    new_partition: dict[T, C] = {}
    n_other = 0
    for member, cid in partition.items():
        if cid in keep:
            new_partition[member] = cid
        else:
            new_partition[member] = other_label
            n_other += 1
    return new_partition, len(keep), n_other
