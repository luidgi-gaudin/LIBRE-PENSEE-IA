"""Checking every default, one at a time, against a ruler that does not move.

The eigenvalue exponent turned out to be wrong. That raised an obvious
question about the settings nobody had checked either, and this answers it
systematically: vary one parameter, hold the rest, score against held-out
similarity.

The methodological point is in `prepare`'s `truth_config`. Several of these
parameters — window, smoothing, pruning — are used both to build the space
and to build the ground truth it is scored against. Rebuilding the truth for
each candidate would move the ruler along with the thing being measured, and
the comparison would be worth nothing. So the vocabulary and the truth table
are computed once, from the defaults, and held fixed for the whole sweep.

A consequence worth stating: `vocab_size` and `min_count` cannot be audited
this way. Changing them changes which words exist, so the pairs and the truth
change too, and there is no fixed ruler left to measure against.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .counts import cooccurrence
from .heldout import (RULER, build_from_tokens, sparse_cosine, spearman,
                      split_blocks)
from .linalg import SparseMatrix
from .pipeline import Config
from .text import Vocabulary, read_tokens
from .weight import ppmi

# Only parameters the fixed ruler can fairly judge.
DEFAULT_GRID: tuple[tuple[str, tuple], ...] = (
    ("window", (2, 4, 6, 10)),
    ("alpha", (0.5, 0.75, 1.0)),
    ("shift", (1.0, 2.0, 5.0)),
    ("min_pair_weight", (0.0, 1.0, 2.0)),
    ("eigenvalue_power", (0.5, 0.75, 1.0, 1.25)),
)


@dataclass
class Ruler:
    """A fixed vocabulary, truth table and pair sample.

    Built once from the default config. Nothing in a sweep is allowed to
    change any of it.
    """

    vocab: Vocabulary
    truth: SparseMatrix
    pairs: list[tuple[int, int]]
    train: list[str]
    test: list[str]

    def score(self, config: Config) -> float:
        """Spearman for a space built with `config`, against this ruler."""
        space, _ = build_from_tokens(self.train, config, verbose=False)
        predicted, actual = [], []
        for i, j in self.pairs:
            truth = sparse_cosine(self.truth.rows[i], self.truth.rows[j])
            if truth == 0.0:
                continue
            predicted.append(
                sum(x * y for x, y in zip(space.units[i], space.units[j]))
            )
            actual.append(truth)
        return spearman(predicted, actual)


def build_ruler(
    paths: list[str],
    base: Config | None = None,
    block: int = 4000,
    pair_vocab: int = 1200,
    pair_count: int = 4000,
    seed: int = 20260811,
) -> Ruler:
    base = base or Config()
    ruler = Config(**{**base.as_dict(), **RULER})

    tokens: list[str] = []
    for path in paths:
        tokens.extend(read_tokens(path))
    train, test = split_blocks(tokens, block=block)

    vocab = Vocabulary.from_tokens(
        train, max_size=base.vocab_size, min_count=base.min_count
    )
    truth = ppmi(
        cooccurrence(
            vocab.encode(test),
            size=len(vocab),
            window=ruler.window,
            harmonic=ruler.harmonic,
            min_weight=ruler.min_pair_weight,
        ),
        alpha=ruler.alpha,
        shift=ruler.shift,
    )

    ceiling = min(pair_vocab, len(vocab))
    rng = random.Random(seed)
    pairs: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < pair_count and attempts < pair_count * 50:
        attempts += 1
        i, j = rng.randrange(ceiling), rng.randrange(ceiling)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in pairs:
            continue
        if not truth.rows[key[0]] or not truth.rows[key[1]]:
            continue
        pairs.add(key)

    return Ruler(
        vocab=vocab,
        truth=truth,
        pairs=sorted(pairs),
        train=train,
        test=test,
    )


@dataclass
class Finding:
    """One parameter, every value tried, and whether the default won."""

    parameter: str
    default: object
    scores: list[tuple[object, float]] = field(default_factory=list)

    @property
    def best(self) -> tuple[object, float]:
        return max(self.scores, key=lambda pair: pair[1])

    @property
    def default_wins(self) -> bool:
        return self.best[0] == self.default

    @property
    def row(self) -> str:
        body = "  ".join(f"{v}={s:.4f}" for v, s in self.scores)
        verdict = "" if self.default_wins else f"   <-- {self.best[0]} beats {self.default}"
        return f"{self.parameter:<17} {body}{verdict}"


def audit(
    ruler: Ruler,
    grid: tuple[tuple[str, tuple], ...] = DEFAULT_GRID,
    base: Config | None = None,
    progress=None,
) -> list[Finding]:
    """Sweep every parameter in the grid, one at a time."""
    base = base or Config()
    cache: dict[tuple, float] = {}
    findings = []

    for parameter, values in grid:
        finding = Finding(parameter=parameter, default=getattr(base, parameter))
        for value in values:
            key = (parameter, value)
            # The default value of every parameter produces the same build,
            # so score it once and reuse it across the whole grid.
            if value == getattr(base, parameter):
                key = ("__default__",)
            if key not in cache:
                if progress:
                    progress(parameter, value)
                cache[key] = ruler.score(
                    Config(**{**base.as_dict(), parameter: value})
                )
            finding.scores.append((value, cache[key]))
        findings.append(finding)
    return findings
