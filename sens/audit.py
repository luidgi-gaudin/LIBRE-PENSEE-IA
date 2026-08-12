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

`vocab_size` and `min_count` need a different fixed ruler, because changing
them changes which words exist. `audit_vocabulary` builds one: it takes the
truth table from the largest vocabulary under test and restricts the scored
pairs to words every candidate contains. That is possible only because
vocabularies from the same corpus are nested — `Vocabulary.from_tokens` sorts
by descending frequency before applying either cut, so both `min_count` and
`max_size` remove a suffix, and a smaller vocabulary is always a prefix of a
larger one with the same word at every shared index.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .counts import cooccurrence
from .heldout import (RULER, build_from_tokens, sparse_cosine, spearman,
                      split_documents)
from .linalg import SparseMatrix
from .pipeline import Config
from .text import Vocabulary, read_tokens
from .weight import ppmi

# Only parameters the fixed ruler can fairly judge.
DEFAULT_GRID: tuple[tuple[str, tuple], ...] = (
    ("window", (2, 4, 6, 10)),
    ("harmonic", (False, True)),
    ("weighting", ("raw", "log", "pmi", "ppmi")),
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

    documents = [read_tokens(path) for path in paths]
    train, test = split_documents(documents, block=block)

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


def audit_vocabulary(
    paths: list[str],
    sizes: tuple[int, ...] = (2000, 4000, 8000),
    min_counts: tuple[int, ...] = (5, 10, 20),
    base: Config | None = None,
    block: int = 4000,
    pair_vocab: int = 1200,
    pair_count: int = 4000,
    seed: int = 20260811,
    progress=None,
) -> list[Finding]:
    """Audit the two parameters that change which words exist.

    The ordinary ruler cannot judge these: a different vocabulary means
    different columns in the truth table and therefore different cosines, so
    every candidate would be marked against a different mark scheme.

    The fix uses the nesting property. Build the truth once from the largest
    vocabulary, then score only word pairs drawn from the first `pair_vocab`
    entries — those are the most frequent words, they exist in every
    candidate, and because the vocabularies are nested they sit at the same
    index in each. The truth never moves; only the space being judged does.
    """
    base = base or Config()
    ruler_cfg = Config(**{**base.as_dict(), **RULER})

    documents = [read_tokens(path) for path in paths]
    train, test = split_documents(documents, block=block)

    reference = Vocabulary.from_tokens(
        train, max_size=max(sizes), min_count=min(min_counts)
    )
    truth = ppmi(
        cooccurrence(
            reference.encode(test),
            size=len(reference),
            window=ruler_cfg.window,
            harmonic=ruler_cfg.harmonic,
            min_weight=ruler_cfg.min_pair_weight,
        ),
        alpha=ruler_cfg.alpha,
        shift=ruler_cfg.shift,
    )

    smallest = min(
        len(Vocabulary.from_tokens(train, max_size=s, min_count=m))
        for s in sizes
        for m in min_counts
    )
    ceiling = min(pair_vocab, smallest)

    rng = random.Random(seed)
    pairs: set[tuple[int, int]] = set()
    attempts = 0
    while len(pairs) < pair_count and attempts < pair_count * 50:
        attempts += 1
        i, j = rng.randrange(ceiling), rng.randrange(ceiling)
        if i == j:
            continue
        key = (min(i, j), max(i, j))
        if key in pairs or not truth.rows[key[0]] or not truth.rows[key[1]]:
            continue
        pairs.add(key)
    ordered = sorted(pairs)

    def score(config: Config) -> float:
        space, _ = build_from_tokens(train, config, verbose=False)
        # The nesting property is load-bearing; check it rather than trust it.
        for index in (0, ceiling - 1):
            if space.words[index] != reference.words[index]:
                raise ValueError(
                    "vocabularies are not nested as assumed: index "
                    f"{index} is {space.words[index]!r} here and "
                    f"{reference.words[index]!r} in the reference"
                )
        predicted, actual = [], []
        for i, j in ordered:
            value = sparse_cosine(truth.rows[i], truth.rows[j])
            if value == 0.0:
                continue
            predicted.append(
                sum(x * y for x, y in zip(space.units[i], space.units[j]))
            )
            actual.append(value)
        return spearman(predicted, actual)

    findings = []
    for parameter, values in (("vocab_size", sizes), ("min_count", min_counts)):
        finding = Finding(parameter=parameter, default=getattr(base, parameter))
        for value in values:
            if progress:
                progress(parameter, value)
            finding.scores.append(
                (value, score(Config(**{**base.as_dict(), parameter: value})))
            )
        findings.append(finding)
    return findings


TOKENISATIONS: tuple[tuple[str, dict], ...] = (
    ("default", {}),
    ("keep case", {"lowercase": False}),
    ("keep accents", {"fold_accents": False}),
    ("split clitics", {"split_clitics": True}),
)


def audit_tokenisation(
    paths: list[str],
    variants: tuple[tuple[str, dict], ...] = TOKENISATIONS,
    base: Config | None = None,
    block: int = 4000,
    pair_vocab: int = 1200,
    pair_count: int = 4000,
    seed: int = 20260811,
    progress=None,
) -> list[tuple[str, float, float]]:
    """Audit the choices made before any counting happens.

    Returns `(label, spearman, coverage)` per variant.

    Tokenisation resists both rulers built so far. It does not merely resize
    the vocabulary, it changes which strings are words at all, so the
    vocabularies are not nested and there is no shared index to lean on.

    What is still shared is the *word*. So the truth table is computed once
    from the default tokenisation, and every candidate is asked about the
    same pairs, looked up by string rather than by position.

    Scoring each variant on whatever pairs it happens to cover is not good
    enough, and the first version of this function did exactly that. A
    variant that drops 15% of the vocabulary is then graded on an easier
    remainder, and "keep accents" came out ten standard deviations *ahead*
    of the default on that basis. So the scored set is the intersection —
    every variant answers the same questions — and coverage is reported
    separately as the cost it is.

    Keeping case is the clearest example. A cased vocabulary holds `the` and
    `The` separately, so the vector this returns for `the` is built from
    only the mid-sentence occurrences. That is not an unfair query; it is
    precisely the cost of not folding.
    """
    base = base or Config()
    ruler_cfg = Config(**{**base.as_dict(), **RULER})

    documents = [read_tokens(path) for path in paths]
    train, test = split_documents(documents, block=block)

    vocab = Vocabulary.from_tokens(
        train, max_size=base.vocab_size, min_count=base.min_count
    )
    truth = ppmi(
        cooccurrence(
            vocab.encode(test),
            size=len(vocab),
            window=ruler_cfg.window,
            harmonic=ruler_cfg.harmonic,
            min_weight=ruler_cfg.min_pair_weight,
        ),
        alpha=ruler_cfg.alpha,
        shift=ruler_cfg.shift,
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
        if key in pairs or not truth.rows[key[0]] or not truth.rows[key[1]]:
            continue
        pairs.add(key)
    ordered = sorted(pairs)

    # Pairs with no held-out evidence are unscorable for everyone.
    evidenced = [
        (i, j) for i, j in ordered
        if sparse_cosine(truth.rows[i], truth.rows[j]) != 0.0
    ]

    spaces = {}
    for label, overrides in variants:
        if progress:
            progress("tokenisation", label)
        config = Config(**{**base.as_dict(), **overrides})
        candidate_docs = [
            read_tokens(
                path,
                lowercase=config.lowercase,
                fold_accents=config.fold_accents,
                split_clitics=config.split_clitics,
            )
            for path in paths
        ]
        candidate_train, _ = split_documents(candidate_docs, block=block)
        spaces[label], _ = build_from_tokens(
            candidate_train, config, verbose=False
        )

    # The common set: pairs every variant can answer. Anything else would
    # grade different variants on different questions.
    common = [
        (i, j) for i, j in evidenced
        if all(
            vocab.words[i] in space and vocab.words[j] in space
            for space in spaces.values()
        )
    ]

    results = []
    for label, _ in variants:
        space = spaces[label]
        covered = sum(
            1 for i, j in evidenced
            if vocab.words[i] in space and vocab.words[j] in space
        )
        predicted, actual = [], []
        for i, j in common:
            predicted.append(
                space.similarity(vocab.words[i], vocab.words[j])
            )
            actual.append(sparse_cosine(truth.rows[i], truth.rows[j]))
        coverage = covered / len(evidenced) if evidenced else 0.0
        results.append((label, spearman(predicted, actual), coverage))
    return results
