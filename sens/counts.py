"""Counting what appears near what.

This is the only place the corpus is actually read for evidence, and the
evidence is thin: a tally of how often word i turned up within a few words
of word j. No syntax, no order beyond distance, no idea what any of it is
about. Everything downstream is a transformation of this table.
"""

from __future__ import annotations

from .linalg import SparseMatrix


def cooccurrence(
    ids: list[int],
    size: int,
    window: int = 4,
    harmonic: bool = True,
    min_weight: float = 0.0,
) -> SparseMatrix:
    """Tally co-occurrences within a symmetric window.

    `harmonic` weights a pair by `1 / distance`, so an immediate neighbour
    counts for one and a word four places away counts for a quarter. This is
    a cheap stand-in for the fact that syntactic relations are mostly local:
    without it, the window's outer edge contributes as much evidence as the
    word right next door, and the whole matrix blurs.

    `min_weight` discards pairs whose total weight never reached the
    threshold. Word pairs are Zipf-distributed too, so the overwhelming
    majority of entries record a single accidental adjacency. Dropping them
    costs almost no signal and can cut the matrix by more than half, which
    matters because the factorisation cost is linear in the number of
    non-zeros.
    """
    rows: list[dict[int, float]] = [{} for _ in range(size)]

    for pos in range(len(ids)):
        i = ids[pos]
        row_i = rows[i]
        start = pos - window
        if start < 0:
            start = 0
        for off in range(start, pos):
            j = ids[off]
            weight = 1.0 / (pos - off) if harmonic else 1.0
            row_i[j] = row_i.get(j, 0.0) + weight
            row_j = rows[j]
            row_j[i] = row_j.get(i, 0.0) + weight

    if min_weight > 0.0:
        rows = [
            {j: w for j, w in row.items() if w >= min_weight} for row in rows
        ]

    return SparseMatrix(rows)
