"""The control this repository needed and did not have.

Every number elsewhere in `sens` measures the compressed space against
something. None of them measured it against *not compressing at all*.

That omission mattered, because the README made a strong claim — that the
factorisation is the step which creates meaning, and that generalisation is
what a lossy encoder does when it runs out of room. A claim like that is only
worth making if the uncompressed matrix has been asked the same questions and
done worse. So this module asks it.

The answer turned out to be more interesting than the claim. Compression does
not win across the board and does not lose across the board. It trades one
thing for another, significantly and in both directions, and the trade is the
actual result.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

from .evaluate import Category, quadruples
from .linalg import SparseMatrix
from .space import Space


class SparseSpace:
    """The uncompressed representation, wearing the same interface as `Space`.

    Rows of the PPMI matrix used directly as word vectors: 4,000 dimensions
    instead of 64, sparse instead of dense, and no factorisation anywhere.
    Everything is normalised once at construction so that a similarity is a
    plain dot product, exactly as in `Space`.
    """

    def __init__(self, words: list[str], matrix: SparseMatrix):
        self.words = words
        self.index = {w: i for i, w in enumerate(words)}
        self.units: list[dict[int, float]] = []
        for row in matrix.rows:
            norm = math.sqrt(sum(v * v for v in row.values()))
            self.units.append(
                {k: v / norm for k, v in row.items()} if norm else {}
            )

    def __len__(self) -> int:
        return len(self.words)

    def __contains__(self, word: str) -> bool:
        return word in self.index

    @staticmethod
    def _dot(a: dict[int, float], b: dict[int, float]) -> float:
        # Iterate whichever row is shorter; the cost of a sparse dot is set
        # by the smaller operand, not the larger.
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b[k] for k, v in a.items() if k in b)

    def similarity(self, a: str, b: str) -> float:
        return self._dot(self.units[self.index[a]], self.units[self.index[b]])

    def analogy(
        self, a: str, b: str, c: str, n: int = 5
    ) -> list[tuple[str, float]]:
        """Vector-offset analogy, computed sparsely."""
        ia, ib, ic = self.index[a], self.index[b], self.index[c]
        target: dict[int, float] = {}
        for k, v in self.units[ib].items():
            target[k] = target.get(k, 0.0) + v
        for k, v in self.units[ia].items():
            target[k] = target.get(k, 0.0) - v
        for k, v in self.units[ic].items():
            target[k] = target.get(k, 0.0) + v

        skip = {ia, ib, ic}
        scored = [
            (self._dot(u, target), i)
            for i, u in enumerate(self.units)
            if i not in skip and u
        ]
        return [(self.words[i], s) for s, i in heapq.nlargest(n, scored)]


def mcnemar(only_a: int, only_b: int) -> float:
    """Two-sided p-value for a paired comparison of two classifiers.

    Both systems answer the same questions, so what carries the evidence is
    not how many each got right but the *disagreements*: questions one solved
    and the other did not. Questions both got right, or both got wrong, say
    nothing about which is better and are discarded.

    Uses the continuity-corrected chi-square with one degree of freedom, for
    which the tail probability is `erfc(sqrt(chi2 / 2))`.
    """
    total = only_a + only_b
    if total == 0:
        return 1.0
    chi2 = (abs(only_a - only_b) - 1) ** 2 / total
    return math.erfc(math.sqrt(chi2 / 2.0))


@dataclass
class Tally:
    """Counts for one representation over one question set."""

    label: str
    asked: int = 0
    top1: int = 0
    top5: int = 0
    form: int = 0

    def rate(self, field_name: str) -> float:
        return getattr(self, field_name) / self.asked if self.asked else 0.0

    @property
    def row(self) -> str:
        return (
            f"{self.label:<28} {self.rate('top1'):7.1%} "
            f"{self.rate('top5'):7.1%} {self.rate('form'):7.1%}"
        )


@dataclass
class Comparison:
    """A paired comparison, with the disagreements that carry the evidence."""

    dense: Tally
    sparse: Tally
    discordant: dict[str, tuple[int, int]] = field(default_factory=dict)

    def significance(self, metric: str) -> tuple[int, int, float]:
        sparse_only, dense_only = self.discordant[metric]
        return sparse_only, dense_only, mcnemar(sparse_only, dense_only)


def compare(
    space: Space,
    sparse: SparseSpace,
    categories: list[Category],
    limit: int = 120,
    seed: int = 20260811,
) -> Comparison:
    """Ask both representations the same questions and pair the answers."""
    dense_tally = Tally("compressed space (64 dims)")
    sparse_tally = Tally("raw PPMI rows (4000 dims)")
    discordant = {"top1": [0, 0], "top5": [0, 0], "form": [0, 0]}

    for category in categories:
        shapes = {derived for _, derived in category.pairs}
        for a, b, c, d in quadruples(category.pairs, limit=limit, seed=seed):
            if not all(w in sparse for w in (a, b, c)):
                continue
            answers = {
                "dense": [w for w, _ in space.analogy(a, b, c, n=5)],
                "sparse": [w for w, _ in sparse.analogy(a, b, c, n=5)],
            }
            outcome = {}
            for which, got in answers.items():
                tally = dense_tally if which == "dense" else sparse_tally
                tally.asked += 1
                hit1 = bool(got) and got[0] == d
                hit5 = d in got
                shape = bool(got) and got[0] in shapes
                tally.top1 += hit1
                tally.top5 += hit5
                tally.form += shape
                outcome[which] = (hit1, hit5, shape)

            for slot, metric in enumerate(("top1", "top5", "form")):
                s = outcome["sparse"][slot]
                dn = outcome["dense"][slot]
                if s and not dn:
                    discordant[metric][0] += 1
                elif dn and not s:
                    discordant[metric][1] += 1

    return Comparison(
        dense=dense_tally,
        sparse=sparse_tally,
        discordant={k: (v[0], v[1]) for k, v in discordant.items()},
    )


def header() -> str:
    return f"{'representation':<28} {'top1':>7} {'top5':>7} {'form':>7}"
