"""Questions the benchmark makes it possible to ask.

Two choices in the pipeline were made by citation rather than by measurement:
the eigenvalue exponent is 0.5 because that is what the literature settles
on, and the negative eigenvalues are kept because ranking by magnitude is
what makes a truncated SVD a truncated SVD. Both are defensible. Neither had
been checked against this corpus.

Now that there is a benchmark, they can be. Both experiments reuse a single
factorisation — the exponent is applied after the eigendecomposition, and an
ablation only drops columns — so a sweep costs one evaluation per setting
instead of one full rebuild per setting.
"""

from __future__ import annotations

import random

from dataclasses import dataclass

from .evaluate import Score, evaluate, totals
from .space import Space

DEFAULT_POWERS = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass
class Result:
    """One configuration, scored."""

    label: str
    dims: int
    score: Score

    @property
    def row(self) -> str:
        return (
            f"{self.label:<22} {self.dims:4}  "
            f"{self.score.accuracy:6.1%}  {self.score.recall5:6.1%}  "
            f"{self.score.form_rate:6.1%}"
        )


def _score(space: Space, method: str, limit: int, seed: int) -> Score:
    scores, _ = evaluate(space, methods=(method,), limit=limit, seed=seed)
    return totals(scores)[method]


def sweep_power(
    space: Space,
    powers: tuple[float, ...] = DEFAULT_POWERS,
    method: str = "3cosadd",
    limit: int = 60,
    seed: int = 20260811,
) -> list[Result]:
    """Score the space at several eigenvalue exponents.

    At power 0 the eigenvalues are discarded entirely and every retained
    direction counts the same. At 1 the vectors are the rows of the rank-64
    reconstruction. 0.5 splits the difference, and is the usual choice
    because it treats the word side and the context side of a symmetric
    factorisation identically.
    """
    results = []
    for power in powers:
        variant = space.rescaled(power)
        results.append(
            Result(
                label=f"power {power:.2f}",
                dims=variant.dim,
                score=_score(variant, method, limit, seed),
            )
        )
    return results


def ablate_signs(
    space: Space,
    method: str = "3cosadd",
    limit: int = 60,
    seed: int = 20260811,
    random_draws: int = 8,
) -> list[Result]:
    """Compare ranking directions by `|eigenvalue|` against ranking by sign.

    Dimension-matching is necessary and, on its own, not sufficient. An
    earlier version of this function compared only

      * the top `p` directions by magnitude, and
      * the `p` positive directions,

    found the first winning by a factor of two, and concluded that the
    negative-eigenvalue directions were carrying signal. That conclusion does
    not survive a control. Dropping the negatives also drops several of the
    *strongest* directions, so the positive-only set is not just
    sign-filtered, it is weakened — and it turns out to score no better than
    `p` axes chosen at random.

    So a random control of the same size is included now, pooled over several
    draws because a single draw is noisy enough to mislead in either
    direction. A configuration only tells you something about sign if it
    departs from random selection at the same dimensionality.

    With the control in place the design decision comes out validated and the
    explanation comes out replaced. Magnitude-ranking beats random. Sign-
    ranking loses to random — not merely ties with it — because selecting for
    positive eigenvalues systematically excludes the largest directions,
    which a random draw would have included in proportion. Magnitude predicts
    how much a direction matters; sign predicts nothing, and selecting on it
    is actively worse than not selecting at all.
    """
    positive = space.positive_dimensions()
    negative = space.negative_dimensions()
    if not negative:
        return []

    count = len(positive)
    # Dimensions come out of the factorisation already ordered by |eigenvalue|,
    # so the top `count` by magnitude is just the first `count` indices.
    results = [
        Result(
            label=f"top {count} by |lambda|",
            dims=count,
            score=_score(space.subspace(list(range(count))), method, limit, seed),
        ),
        Result(
            label=f"positive only ({count})",
            dims=count,
            score=_score(space.subspace(positive), method, limit, seed),
        ),
    ]
    # Pool the random draws into one row. Each draw answers the same
    # questions, so summing the counts gives the mean rate with `draws` times
    # the sample behind it, which is what makes the comparison readable.
    pooled = Score(method=method, category="random", asked=0, top1=0, top5=0)
    for draw in range(random_draws):
        picked = sorted(
            random.Random(seed + draw).sample(range(space.dim), count)
        )
        one = _score(space.subspace(picked), method, limit, seed)
        pooled.asked += one.asked
        pooled.top1 += one.top1
        pooled.top5 += one.top5
        pooled.form += one.form
    results.append(
        Result(
            label=f"random {count} ({random_draws} draws)",
            dims=count,
            score=pooled,
        )
    )
    results.append(
        Result(
            label=f"all {space.dim}",
            dims=space.dim,
            score=_score(space, method, limit, seed),
        )
    )
    return results


@dataclass
class DimensionRow:
    """One dimensionality, scored on both metrics at once."""

    dims: int
    heldout: float
    top5: float
    form: float

    @property
    def row(self) -> str:
        return (
            f"{self.dims:5}  {self.heldout:12.4f}  "
            f"{self.top5:7.1%}  {self.form:7.1%}"
        )


def dimension_curve(
    prepared,
    dims: tuple[int, ...] = (16, 32, 48, 64, 96, 128, 160),
    limit: int = 60,
    seed: int = 20260811,
) -> list[DimensionRow]:
    """Score a range of dimensionalities on similarity and on category.

    Takes a `heldout.Prepared` built at the largest dimension in `dims`;
    every smaller one is a prefix of it, because the factorisation returns
    directions ordered by `|eigenvalue|`. So the whole curve costs one build.

    The two metrics disagree, and the disagreement is the point. Held-out
    similarity keeps improving with more dimensions. The form rate — how
    often the answer is at least the right *kind* of word — peaks early and
    then falls back toward what the uncompressed matrix scores. Adding axes
    buys detail and spends category.
    """
    from .evaluate import evaluate as run_benchmark, totals
    from .heldout import correlation

    rows = []
    for k in dims:
        variant = prepared.space.subspace(list(range(k)))
        rho, _ = correlation(variant, prepared)
        scores, _ = run_benchmark(
            variant, methods=("3cosadd",), limit=limit, seed=seed
        )
        tally = totals(scores)["3cosadd"]
        rows.append(
            DimensionRow(
                dims=k,
                heldout=rho,
                top5=tally.recall5,
                form=tally.form_rate,
            )
        )
    return rows


def dimension_header() -> str:
    return f"{'dims':>5}  {'held-out':>12}  {'top5':>7}  {'form':>7}"


def header() -> str:
    return f"{'configuration':<22} {'dims':>4}  {'top1':>6}  {'top5':>6}  {'form':>6}"
