"""How big does a difference have to be before it means anything?

Almost every claim in this repository is one number against another. The
exponent moved because 0.6002 beat 0.5077; `alpha` moved because 0.6002 beat
0.5908. Those two comparisons look alike written down and are not alike at
all, and nothing here could tell them apart, because no measurement had an
error bar.

This supplies one. The factorisation starts from a random block, so the seed
changes the answer while changing nothing about the method. Rebuilding the
same configuration under several seeds and looking at the spread gives the
noise floor: the size of a difference that means nothing.

It is deliberately the crudest possible estimate. The seed is only one source
of variation — the corpus is fixed, the benchmark questions are fixed, the
split is fixed — so the real uncertainty on any number here is larger than
what this reports, never smaller. That is the useful direction to be wrong
in: a difference this calls noise is definitely noise.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field

from .audit import Ruler
from .evaluate import evaluate, totals
from .pipeline import Config, build

DEFAULT_SEEDS = (20260811, 7, 42, 1234, 99999, 31415)


@dataclass
class Spread:
    """One metric, measured several times under identical settings."""

    metric: str
    percent: bool
    values: list[float] = field(default_factory=list)

    @property
    def mean(self) -> float:
        return statistics.mean(self.values) if self.values else 0.0

    @property
    def sd(self) -> float:
        return statistics.stdev(self.values) if len(self.values) > 1 else 0.0

    @property
    def floor(self) -> float:
        """Two standard deviations: the smallest difference worth reporting."""
        return 2.0 * self.sd

    def format(self, value: float) -> str:
        return f"{value:.1%}" if self.percent else f"{value:.4f}"

    @property
    def row(self) -> str:
        return (
            f"{self.metric:<10} mean {self.format(self.mean):>8}  "
            f"sd {self.format(self.sd):>8}  "
            f"range {self.format(min(self.values))}-{self.format(max(self.values))}"
            f"   noise floor {self.format(self.floor)}"
        )


def measure(
    paths: list[str],
    ruler: Ruler,
    seeds: tuple[int, ...] = DEFAULT_SEEDS,
    limit: int = 120,
    progress=None,
) -> list[Spread]:
    """Rebuild under several seeds and report the spread of each metric.

    Only the build seed changes. The corpus, the vocabulary, the held-out
    split, the benchmark questions and every other parameter are identical
    across runs, so whatever spread comes out is attributable to the random
    start of the factorisation and nothing else.
    """
    spreads = {
        "held-out": Spread("held-out", percent=False),
        "top1": Spread("top1", percent=True),
        "top5": Spread("top5", percent=True),
        "form": Spread("form", percent=True),
    }

    for seed in seeds:
        if progress:
            progress(seed)
        config = Config(seed=seed)
        spreads["held-out"].values.append(ruler.score(config))
        space, _ = build(paths, config, verbose=False)
        tallies = totals(evaluate(space, methods=("3cosadd",), limit=limit)[0])
        if "3cosadd" not in tallies:
            raise ValueError(
                "the vocabulary produced no benchmark questions, so there is "
                "nothing to measure the spread of; this needs a corpus large "
                "enough for sens.evaluate.build_benchmark to find word pairs"
            )
        tally = tallies["3cosadd"]
        spreads["top1"].values.append(tally.accuracy)
        spreads["top5"].values.append(tally.recall5)
        spreads["form"].values.append(tally.form_rate)

    return list(spreads.values())


def verdict(spread: Spread, difference: float) -> str:
    """Describe a difference in units of the noise floor."""
    if spread.sd <= 0.0:
        return "no spread measured"
    sigma = abs(difference) / spread.sd
    if sigma < 2.0:
        return f"{sigma:.1f} sd — within noise"
    if sigma < 3.0:
        return f"{sigma:.1f} sd — marginal"
    return f"{sigma:.1f} sd — solid"
