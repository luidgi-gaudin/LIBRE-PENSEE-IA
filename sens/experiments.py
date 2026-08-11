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
) -> list[Result]:
    """Ask whether the negative-eigenvalue directions carry anything.

    The comparison has to be dimension-matched or it proves nothing: a space
    with fewer axes scores worse for reasons that have nothing to do with
    which axes were dropped. So if `p` of the retained directions have
    positive eigenvalues, this compares

      * the `p` positive directions, and
      * the top `p` directions by magnitude, negatives included,

    both of which are `p`-dimensional. If magnitude-ranking wins, the
    negative eigenvalues were carrying signal and discarding them would be a
    real loss. If it does not, keeping them was theory rather than benefit.
    """
    positive = space.positive_dimensions()
    negative = space.negative_dimensions()
    if not negative:
        return []

    count = len(positive)
    # Dimensions come out of the factorisation already ordered by |eigenvalue|,
    # so the top `count` by magnitude is just the first `count` indices.
    by_magnitude = list(range(count))

    return [
        Result(
            label=f"top {count} by |lambda|",
            dims=count,
            score=_score(space.subspace(by_magnitude), method, limit, seed),
        ),
        Result(
            label=f"positive only ({count})",
            dims=count,
            score=_score(space.subspace(positive), method, limit, seed),
        ),
        Result(
            label=f"negative only ({len(negative)})",
            dims=len(negative),
            score=_score(space.subspace(negative), method, limit, seed),
        ),
        Result(
            label=f"all {space.dim}",
            dims=space.dim,
            score=_score(space, method, limit, seed),
        ),
    ]


def header() -> str:
    return f"{'configuration':<22} {'dims':>4}  {'top1':>6}  {'top5':>6}  {'form':>6}"
