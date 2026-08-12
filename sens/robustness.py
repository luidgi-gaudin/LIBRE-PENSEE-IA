"""How much of the confidence in this repository is real?

`sens noise` measures one source of variation: the random block the
factorisation starts from. Every standard-deviation figure here is quoted
against it, which quietly assumes that the seed is the only thing that could
have come out differently.

It is not. The held-out measurement rests on a stack of choices nobody made
deliberately:

* which order the corpus files happen to be listed in,
* how big a block the train/test split deals in,
* where the first block boundary falls,
* which word pairs the sample happens to draw.

None of these is a hypothesis. All of them could have been otherwise. The
first was found by accident, when the same corpus in a different order moved
results by five to six standard deviations — more than the seed does. That
one is now fixed. The others were never checked, and there is no reason to
believe the first was special.

So this varies each of them in turn and reports what it costs. The number
that matters is not any individual spread but their combination: if the real
uncertainty is twice the seed-only floor, then every `sd` in this README is
overstated by the same factor, and some claims move from solid to marginal.

That would be an unwelcome result and is the reason to measure it.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass, field

from .audit import build_ruler
from .pipeline import Config


@dataclass
class Axis:
    """One incidental choice, and the spread it turns out to be worth."""

    name: str
    what: str
    settings: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)

    @property
    def sd(self) -> float:
        return statistics.stdev(self.scores) if len(self.scores) > 1 else 0.0

    @property
    def spread(self) -> float:
        return max(self.scores) - min(self.scores) if self.scores else 0.0

    @property
    def row(self) -> str:
        return (
            f"{self.name:<16} {self.sd:8.4f} {self.spread:9.4f}   "
            f"{', '.join(self.settings)}"
        )


def probe(
    paths: list[str],
    seeds: tuple[int, ...] = (20260811, 7, 42),
    blocks: tuple[int, ...] = (2000, 4000, 8000),
    pair_seeds: tuple[int, ...] = (20260811, 7, 42, 1234, 99999, 31415),
    progress=None,
) -> list[Axis]:
    """Vary each incidental choice in turn and score the default build.

    Absolute held-out scores, not differences. A difference between two
    builds cancels part of whatever these axes do to both of them, so the
    spread reported here is an upper bound on what a claim actually suffers.
    An upper bound is the useful direction to err in: an axis this calls
    harmless is harmless.
    """
    base = Config()
    axes: list[Axis] = []

    factorisation = Axis(
        "factorisation", "the random block the factorisation starts from"
    )
    ruler = build_ruler(paths)
    for seed in seeds:
        if progress:
            progress("factorisation", seed)
        factorisation.settings.append(str(seed))
        factorisation.scores.append(
            ruler.score(Config(**{**base.as_dict(), "seed": seed}))
        )
    axes.append(factorisation)

    pairs = Axis("pair sample", "which word pairs the ruler happens to draw")
    for seed in pair_seeds:
        if progress:
            progress("pair sample", seed)
        # Only the sample moves; the space and the truth table are the ones
        # already built, so this is nearly free.
        sampled = build_ruler(paths, seed=seed)
        pairs.settings.append(str(seed))
        pairs.scores.append(sampled.score(base))
    axes.append(pairs)

    size = Axis("block size", "how many tokens the split deals at a time")
    for block in blocks:
        if progress:
            progress("block size", block)
        sized = build_ruler(paths, block=block)
        size.settings.append(str(block))
        size.scores.append(sized.score(base))
    axes.append(size)

    return axes


def probe_analogy(
    space,
    question_seeds: tuple[int, ...] = (20260811, 7, 42, 1234, 99999, 31415),
    limit: int = 120,
) -> list[Axis]:
    """The same question, asked of the analogy benchmark.

    The benchmark draws its questions with a fixed seed, which is the exact
    counterpart of the pair sample on the held-out side, and turned out to
    be the same size of problem: `form` moves twice as much with the
    question sample as with the factorisation seed.

    One caveat matters more than the numbers. A **paired** comparison —
    McNemar over two systems answering the identical question set — is
    immune to this entirely, because a question that is hard is hard for
    both sides and drops out of the disagreement count. Every p-value in
    this repository is paired and survives untouched. What does not survive
    is an effect quoted as a number of points against a floor that only
    counted rebuilds.
    """
    from .evaluate import evaluate, totals

    axes = {
        name: Axis(f"questions/{name}", "which questions the benchmark draws")
        for name in ("top1", "top5", "form")
    }
    for seed in question_seeds:
        tally = totals(
            evaluate(space, methods=("3cosadd",), limit=limit, seed=seed)[0]
        )["3cosadd"]
        for name, value in (("top1", tally.accuracy),
                            ("top5", tally.recall5),
                            ("form", tally.form_rate)):
            axes[name].settings.append(str(seed))
            axes[name].scores.append(value)
    return list(axes.values())


def combined(axes: list[Axis]) -> float:
    """Total uncertainty, treating the axes as independent.

    Independence is an assumption and probably a generous one, but the
    alternative — quoting the largest single axis — understates a stack of
    choices that genuinely all vary at once.
    """
    return math.sqrt(sum(a.sd ** 2 for a in axes))


def header() -> str:
    return f"{'axis':<16} {'sd':>8} {'spread':>9}   settings"


def verdict(axes: list[Axis], seed_only: float) -> str:
    total = combined(axes)
    if seed_only <= 0:
        return "no seed-only floor to compare against"
    factor = total / seed_only
    return (
        f"total {total:.4f} against a seed-only floor of {seed_only:.4f} "
        f"— every sd quoted in this repository is optimistic by {factor:.1f}x"
    )
