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
    alpha: float = 1.0,
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
    value word2vec's negative sampler uses, arrived at empirically.

    It does nothing here, and the story of finding that out is instructive.
    `sens audit` first put 1.0 ahead of 0.75 by 2.4 standard deviations, and
    the default moved on that. Re-derived after the held-out split was made
    independent of file ordering, the same comparison is 0.9 sd — inside the
    noise. The two settings are indistinguishable on this corpus. 1.0 remains
    the default because it is the simpler of two equals, not because it wins.

    An explanation was offered and killed along the way: that the vocabulary
    cap has already removed the rare tail smoothing exists to tame. That
    predicts the effect reverses with a bigger vocabulary. Raising the cap to
    12,000 words with a floor of three occurrences did not reverse it.

    `shift` subtracts `log(shift)` from every score before clipping, which is
    the explicit form of skip-gram's negative sampling count. Raising it
    sparsifies the matrix by demanding stronger evidence. At 1.0 it does
    nothing, which is the default.
    """
    return _pmi(counts, alpha=alpha, shift=shift, clip=True)


def _pmi(
    counts: SparseMatrix,
    alpha: float,
    shift: float,
    clip: bool,
) -> SparseMatrix:
    """The shared body of `ppmi` and `pmi`.

    One implementation with a flag, rather than two that can drift apart —
    the whole point of comparing them is that they differ in exactly one
    respect, and two copies of this arithmetic would eventually differ in
    more.
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
            if value > 0.0 or not clip:
                kept[j] = value
        out.append(kept)

    return SparseMatrix(out)


def raw(counts: SparseMatrix, **_) -> SparseMatrix:
    """No weighting at all: the co-occurrence table, untouched.

    The control for the whole third stage. This README claims that turning
    counts into surprise is the only step that makes a claim about language;
    a claim like that is worth nothing until the version without it has been
    asked the same questions.
    """
    return SparseMatrix([dict(row) for row in counts.rows])


def log_counts(counts: SparseMatrix, **_) -> SparseMatrix:
    """`log(1 + count)`, squashing the frequency range and nothing else.

    Between `raw` and `ppmi` in ambition. It fixes the complaint that a
    handful of enormous counts dominate every cosine, without ever comparing
    a pair against what independence would predict. If most of PPMI's value
    is really just compressing a Zipfian range, this recovers it; if the
    value is in the comparison, this does not.
    """
    return SparseMatrix([
        {j: math.log1p(v) for j, v in row.items() if v > 0.0}
        for row in counts.rows
    ])


def pmi(
    counts: SparseMatrix,
    alpha: float = 1.0,
    shift: float = 1.0,
) -> SparseMatrix:
    """Pointwise mutual information with the negatives kept.

    Identical to `ppmi` except that scores below zero survive. The clipping
    in `ppmi` is argued for in the README on the grounds that a negative
    score is mostly noise at this corpus size — an argument that needs the
    unclipped version to be measurable before it means anything.
    """
    return _pmi(counts, alpha=alpha, shift=shift, clip=False)


WEIGHTINGS = {
    "ppmi": ppmi,
    "pmi": pmi,
    "log": log_counts,
    "raw": raw,
}
