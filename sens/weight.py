"""Turning counts into surprise.

A raw count says almost nothing. `the` sits next to everything, so a large
count with `the` is not evidence of a relationship — it is evidence that
`the` is common. What carries information is the gap between how often two
words actually co-occur and how often they would co-occur if they were
independent. That ratio is pointwise mutual information, and it is the
step where the table stops being a census and starts being a claim.
"""

from __future__ import annotations

import math

from .linalg import SparseMatrix


def ppmi(
    counts: SparseMatrix,
    alpha: float = 0.75,
    shift: float = 1.0,
) -> SparseMatrix:
    """Positive pointwise mutual information, with context smoothing.

        pmi(i, j) = log( p(i, j) / ( p(i) * p_alpha(j) ) )

    and everything negative is clipped to zero. The clipping is not tidiness:
    a negative PMI asserts that two words avoid each other, and at corpus
    scales like this one, most such assertions are noise about pairs that
    simply never had the chance to meet. Keeping only positive evidence
    throws away a real signal to avoid a much larger imaginary one.

    `alpha` raises the context probabilities to a fractional power before
    normalising. This flattens the context distribution, which reduces the
    bonus that rare contexts get for being rare. Without it, the highest PMI
    scores in any corpus belong to typos and hapax legomena. 0.75 is the
    value word2vec's negative sampler uses, arrived at empirically, and it
    transfers here for the same reason it worked there.

    `shift` subtracts `log(shift)` from every score before clipping, which is
    the explicit form of skip-gram's negative sampling count. Raising it
    sparsifies the matrix by demanding stronger evidence. At 1.0 it does
    nothing, which is the default.
    """
    n = counts.n
    marginal = [sum(row.values()) for row in counts.rows]
    total = sum(marginal)
    if total <= 0.0:
        return SparseMatrix([{} for _ in range(n)])

    smoothed = [m**alpha for m in marginal]
    smoothed_total = sum(smoothed)

    # The two normalising totals cancel out of the ratio, so all that is left
    # is a per-row offset and a per-column offset. Precomputing both turns the
    # inner loop into one logarithm and two subtractions.
    log_shift = math.log(shift) if shift > 0 else 0.0
    row_offset = [math.log(m) if m > 0.0 else 0.0 for m in marginal]
    col_offset = [
        math.log(s / smoothed_total) if s > 0.0 else 0.0 for s in smoothed
    ]

    out: list[dict[int, float]] = []
    for i, row in enumerate(counts.rows):
        if marginal[i] <= 0.0:
            out.append({})
            continue
        base = -row_offset[i] - log_shift
        kept: dict[int, float] = {}
        for j, c in row.items():
            if c <= 0.0:
                continue
            value = math.log(c) + base - col_offset[j]
            if value > 0.0:
                kept[j] = value
        out.append(kept)

    return SparseMatrix(out)
