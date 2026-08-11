"""Turning an impression into a number.

Nearest-neighbour lists are easy to admire and impossible to argue with. You
read `whale -> sperm, fishery, whales`, you nod, and you have learned nothing
you could compare against another build. This module exists so that claims
about the space can be wrong.

The hard part of evaluating a distributional model is normally the test set:
somebody has to decide, by hand, which words are similar. That data does not
exist for a corpus of six novels, and importing WordSim-353 would both break
the no-dependencies rule and ask questions about a vocabulary this corpus
does not have.

So the benchmark is generated from the vocabulary itself. English morphology
is regular enough that `walk : walked :: work : worked` can be constructed by
string manipulation, with no human judgement and no download. The relation is
real, the model was never told about it, and whether the geometry encodes it
is exactly the sort of question a number can settle.

What this measures is morphological analogy. It is not a proxy for semantic
analogy, and a model could score well here while knowing nothing about
meaning. It is the part that can be measured honestly at this scale.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from .space import Space


@dataclass(frozen=True)
class Rule:
    """A regular derivation, as a prefix and a suffix to bolt onto a stem."""

    name: str
    suffix: str = ""
    prefix: str = ""
    min_stem: int = 4

    def derive(self, stem: str) -> str:
        return self.prefix + stem + self.suffix


# Only fully regular derivations are here. Irregulars (`go`/`went`,
# `good`/`better`) would need a hand-written list, and a hand-written list is
# the thing this module is built to avoid.
RULES: tuple[Rule, ...] = (
    Rule("plural", suffix="s"),
    Rule("past", suffix="ed"),
    Rule("progressive", suffix="ing"),
    Rule("adverb", suffix="ly"),
    Rule("possessive", suffix="'s"),
    Rule("er-form", suffix="er"),
    Rule("negation", prefix="un"),
)


@dataclass
class Category:
    """One relation, and the word pairs in the vocabulary that show it."""

    rule: Rule
    pairs: list[tuple[str, str]] = field(default_factory=list)

    @property
    def name(self) -> str:
        return self.rule.name


def find_pairs(space: Space, rule: Rule) -> list[tuple[str, str]]:
    """Every `(stem, derived)` pair where both forms made the vocabulary.

    The vocabulary is the filter that makes this work without a dictionary.
    A bad stem produces a derived form that is not a word, and a form that is
    not a word is almost never frequent enough to be in the top 4,000. What
    survives is mostly genuine.

    Mostly, not entirely: `on`/`ones` would pass if the stem minimum let it,
    which is what the minimum is for. Some noise remains and is reported
    rather than hidden, because a benchmark that quietly launders its own
    test items is worse than no benchmark.
    """
    pairs = []
    for stem in space.words:
        if len(stem) < rule.min_stem:
            continue
        # A stem that already ends in its own suffix produces doubling
        # artifacts (`sees` -> `seess`) and near-duplicates that test nothing.
        if rule.suffix and stem.endswith(rule.suffix):
            continue
        if rule.prefix and stem.startswith(rule.prefix):
            continue
        derived = rule.derive(stem)
        if derived in space:
            pairs.append((stem, derived))
    return pairs


def build_benchmark(
    space: Space, min_pairs: int = 4
) -> list[Category]:
    """Collect every relation the vocabulary can demonstrate."""
    categories = []
    for rule in RULES:
        pairs = find_pairs(space, rule)
        if len(pairs) >= min_pairs:
            categories.append(Category(rule=rule, pairs=pairs))
    return categories


def quadruples(
    pairs: list[tuple[str, str]], limit: int, seed: int
) -> list[tuple[str, str, str, str]]:
    """Draw `a : b :: c : d` questions from a list of pairs.

    Every ordered pair of pairs is a valid question, so the full set grows
    quadratically and is mostly redundant. Sampling from a sorted list with a
    fixed seed keeps the benchmark both small and identical between runs —
    a benchmark that moves under you cannot show you a regression.
    """
    ordered = sorted(pairs)
    everything = [
        (a, b, c, d)
        for i, (a, b) in enumerate(ordered)
        for j, (c, d) in enumerate(ordered)
        if i != j
    ]
    if len(everything) <= limit:
        return everything
    return random.Random(seed).sample(everything, limit)


@dataclass
class Score:
    """How one method did on one category.

    `form` is the interesting column. It counts answers that are the right
    *kind* of word even when they are the wrong word: asked for `bingley's`
    the model says `darcy's`, which is a possessive belonging to the wrong
    person. Accuracy alone cannot tell that apart from `darcy`, and the two
    failures mean opposite things. One says the relation is missing from the
    geometry; the other says the relation is there and the identity leaked.
    """

    method: str
    category: str
    asked: int
    top1: int
    top5: int
    form: int = 0

    @property
    def accuracy(self) -> float:
        return self.top1 / self.asked if self.asked else 0.0

    @property
    def recall5(self) -> float:
        return self.top5 / self.asked if self.asked else 0.0

    @property
    def form_rate(self) -> float:
        return self.form / self.asked if self.asked else 0.0


def score_category(
    space: Space,
    category: Category,
    method: str,
    limit: int,
    seed: int,
) -> tuple[Score, list[tuple[str, str, str, str, str]]]:
    """Run one category and return its score plus every miss.

    The misses are returned rather than counted, because on a corpus this
    small the interesting question is not how often the model is wrong but
    what it says instead.
    """
    solve = space.analogy_mul if method == "3cosmul" else space.analogy
    questions = quadruples(category.pairs, limit=limit, seed=seed)
    # Every derived form this category knows about. An answer inside this set
    # has the right shape whether or not it has the right stem.
    shapes = {derived for _, derived in category.pairs}

    top1 = top5 = form = 0
    misses = []
    for a, b, c, d in questions:
        answers = [w for w, _ in solve(a, b, c, n=5)]
        if answers and answers[0] == d:
            top1 += 1
        elif answers:
            misses.append((a, b, c, d, answers[0]))
        if answers and answers[0] in shapes:
            form += 1
        if d in answers:
            top5 += 1

    score = Score(
        method=method,
        category=category.name,
        asked=len(questions),
        top1=top1,
        top5=top5,
        form=form,
    )
    return score, misses


def evaluate(
    space: Space,
    methods: tuple[str, ...] = ("3cosadd", "3cosmul"),
    limit: int = 120,
    seed: int = 20260811,
) -> tuple[list[Score], dict[str, list]]:
    """Score every method on every category the vocabulary supports."""
    categories = build_benchmark(space)
    scores: list[Score] = []
    misses: dict[str, list] = {}
    for category in categories:
        for method in methods:
            score, missed = score_category(
                space, category, method, limit=limit, seed=seed
            )
            scores.append(score)
            misses[f"{method}/{category.name}"] = missed
    return scores, misses


def totals(scores: list[Score]) -> dict[str, Score]:
    """Roll per-category scores up into one line per method."""
    combined: dict[str, Score] = {}
    for score in scores:
        running = combined.get(score.method)
        if running is None:
            combined[score.method] = Score(
                method=score.method,
                category="ALL",
                asked=score.asked,
                top1=score.top1,
                top5=score.top5,
                form=score.form,
            )
        else:
            running.asked += score.asked
            running.top1 += score.top1
            running.top5 += score.top5
            running.form += score.form
    return combined
