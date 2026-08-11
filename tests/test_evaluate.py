"""Tests for the benchmark generator.

A benchmark is a measuring instrument, and an instrument that is wrong in
its own favour is worse than none at all. These tests care mostly about two
things: that the generated questions are actually well formed, and that the
scoring counts what it says it counts.
"""

from __future__ import annotations

import unittest

from sens.evaluate import (
    Category,
    Rule,
    Score,
    build_benchmark,
    evaluate,
    find_pairs,
    quadruples,
    score_category,
    totals,
)
from sens.space import Space


def vocabulary(words: list[str]) -> Space:
    """A space whose geometry is junk but whose word list is what we want."""
    return Space(
        words=words,
        vectors=[[float(i + 1), 1.0] for i in range(len(words))],
    )


def parallelogram() -> tuple[Space, Category]:
    """A space where the plural relation is exactly a translation.

    Each stem gets its own axis, and every plural is its stem plus a shared
    final axis. The analogy answers are then a matter of arithmetic, so a
    correct scorer must report 100% here and anything less is a bug in the
    scorer rather than a limitation of the model.
    """
    stems = ["walk", "talk", "jump", "look"]
    words, vectors = [], []
    for i, stem in enumerate(stems):
        base = [0.0] * (len(stems) + 1)
        base[i] = 1.0
        words.append(stem)
        vectors.append(base)
        plural = list(base)
        plural[-1] = 1.0
        words.append(stem + "s")
        vectors.append(plural)
    space = Space(words=words, vectors=vectors)
    category = Category(
        rule=Rule("plural", suffix="s"),
        pairs=[(s, s + "s") for s in stems],
    )
    return space, category


class TestRule(unittest.TestCase):
    def test_suffix_derivation(self):
        self.assertEqual(Rule("plural", suffix="s").derive("ship"), "ships")

    def test_prefix_derivation(self):
        self.assertEqual(Rule("neg", prefix="un").derive("able"), "unable")

    def test_both_at_once(self):
        rule = Rule("x", prefix="un", suffix="ly")
        self.assertEqual(rule.derive("happi"), "unhappily")


class TestFindPairs(unittest.TestCase):
    def test_finds_a_pair_when_both_forms_exist(self):
        space = vocabulary(["ship", "ships"])
        self.assertEqual(
            find_pairs(space, Rule("plural", suffix="s")), [("ship", "ships")]
        )

    def test_ignores_a_stem_whose_derived_form_is_absent(self):
        space = vocabulary(["ship", "whale"])
        self.assertEqual(find_pairs(space, Rule("plural", suffix="s")), [])

    def test_short_stems_are_rejected(self):
        # `hi` + `s` = `his`, a real word and a fake pair. The stem minimum
        # is the only thing standing between the benchmark and that noise.
        space = vocabulary(["hi", "his"])
        self.assertEqual(find_pairs(space, Rule("plural", suffix="s")), [])

    def test_stem_minimum_is_configurable(self):
        space = vocabulary(["hi", "his"])
        rule = Rule("plural", suffix="s", min_stem=2)
        self.assertEqual(find_pairs(space, rule), [("hi", "his")])

    def test_a_stem_already_carrying_the_suffix_is_skipped(self):
        space = vocabulary(["glass", "glasses", "glasss"])
        pairs = find_pairs(space, Rule("plural", suffix="s"))
        self.assertNotIn(("glass", "glasss"), pairs)

    def test_a_stem_already_carrying_the_prefix_is_skipped(self):
        space = vocabulary(["under", "ununder"])
        self.assertEqual(find_pairs(space, Rule("neg", prefix="un")), [])

    def test_results_follow_vocabulary_order(self):
        space = vocabulary(["whale", "whales", "ship", "ships"])
        self.assertEqual(
            [stem for stem, _ in find_pairs(space, Rule("p", suffix="s"))],
            ["whale", "ship"],
        )


class TestBuildBenchmark(unittest.TestCase):
    def test_drops_categories_with_too_few_pairs(self):
        space = vocabulary(["ship", "ships"])
        self.assertEqual(build_benchmark(space, min_pairs=4), [])

    def test_keeps_a_category_that_clears_the_bar(self):
        words = []
        for stem in ["ship", "whale", "boat", "sail", "rope"]:
            words += [stem, stem + "s"]
        categories = build_benchmark(vocabulary(words), min_pairs=4)
        self.assertEqual([c.name for c in categories], ["plural"])
        self.assertEqual(len(categories[0].pairs), 5)

    def test_empty_vocabulary_yields_no_categories(self):
        self.assertEqual(build_benchmark(vocabulary(["a", "b"])), [])


class TestQuadruples(unittest.TestCase):
    def setUp(self):
        self.pairs = [("a", "as"), ("b", "bs"), ("c", "cs"), ("d", "ds")]

    def test_uses_every_ordered_combination_when_under_the_limit(self):
        # 4 pairs give 4 * 3 = 12 ordered questions.
        self.assertEqual(len(quadruples(self.pairs, limit=99, seed=1)), 12)

    def test_never_asks_a_pair_about_itself(self):
        for a, b, c, d in quadruples(self.pairs, limit=99, seed=1):
            self.assertNotEqual((a, b), (c, d))

    def test_questions_are_well_formed(self):
        lookup = dict(self.pairs)
        for a, b, c, d in quadruples(self.pairs, limit=99, seed=1):
            self.assertEqual(lookup[a], b)
            self.assertEqual(lookup[c], d)

    def test_limit_is_respected(self):
        self.assertEqual(len(quadruples(self.pairs, limit=5, seed=1)), 5)

    def test_same_seed_gives_the_same_questions(self):
        first = quadruples(self.pairs, limit=5, seed=7)
        second = quadruples(self.pairs, limit=5, seed=7)
        self.assertEqual(first, second)

    def test_different_seeds_give_different_questions(self):
        first = quadruples(self.pairs, limit=5, seed=7)
        second = quadruples(self.pairs, limit=5, seed=8)
        self.assertNotEqual(first, second)

    def test_input_order_does_not_matter(self):
        # The pairs are sorted internally, so a shuffled input must produce
        # an identical benchmark. Otherwise the score would depend on the
        # order words happened to appear in the corpus.
        shuffled = list(reversed(self.pairs))
        self.assertEqual(
            quadruples(self.pairs, limit=5, seed=3),
            quadruples(shuffled, limit=5, seed=3),
        )


class TestScore(unittest.TestCase):
    def test_rates(self):
        score = Score("m", "c", asked=8, top1=2, top5=4, form=6)
        self.assertAlmostEqual(score.accuracy, 0.25)
        self.assertAlmostEqual(score.recall5, 0.5)
        self.assertAlmostEqual(score.form_rate, 0.75)

    def test_no_questions_is_zero_not_a_crash(self):
        score = Score("m", "c", asked=0, top1=0, top5=0)
        self.assertEqual(score.accuracy, 0.0)
        self.assertEqual(score.recall5, 0.0)
        self.assertEqual(score.form_rate, 0.0)


class TestScoreCategory(unittest.TestCase):
    def test_a_perfect_space_scores_perfectly(self):
        space, category = parallelogram()
        for method in ("3cosadd", "3cosmul"):
            score, misses = score_category(
                space, category, method, limit=99, seed=1
            )
            self.assertEqual(score.asked, 12, method)
            self.assertEqual(score.top1, 12, method)
            self.assertEqual(score.top5, 12, method)
            self.assertEqual(misses, [], method)

    def test_a_perfect_space_is_also_perfect_on_form(self):
        space, category = parallelogram()
        score, _ = score_category(space, category, "3cosadd", 99, 1)
        self.assertEqual(score.form, score.asked)

    def test_form_counts_right_shape_wrong_stem(self):
        # A synthetic version of the failure this metric exists for. All
        # three plurals sit in a tight cluster on their own axis, and their
        # small perturbations point at the wrong stems. So the model always
        # answers with a plural and usually picks the wrong one: form should
        # be perfect while accuracy is not.
        space = Space(
            words=["alpha", "alphas", "gamma", "gammas", "delta", "deltas"],
            vectors=[
                [1.0, 0.0, 0.0, 0.0], [0.05, 0.0, 0.00, 1.0],
                [0.0, 1.0, 0.0, 0.0], [0.00, 0.0, 0.05, 1.0],
                [0.0, 0.0, 1.0, 0.0], [0.00, 0.05, 0.0, 1.0],
            ],
        )
        category = Category(
            rule=Rule("plural", suffix="s"),
            pairs=[("alpha", "alphas"), ("gamma", "gammas"),
                   ("delta", "deltas")],
        )
        for method in ("3cosadd", "3cosmul"):
            score, misses = score_category(space, category, method, 99, 1)
            self.assertEqual(score.asked, 6, method)
            self.assertEqual(score.form, 6, method)
            self.assertLess(score.top1, score.form, method)
            # Every wrong answer is still a plural — that is the whole point.
            shapes = {b for _, b in category.pairs}
            for *_, got in misses:
                self.assertIn(got, shapes, method)

    def test_misses_report_the_answer_that_was_given(self):
        space, category = parallelogram()
        # Ask with a method name that is not 3cosmul, on a broken space.
        broken = Space(
            words=space.words,
            vectors=[[1.0, 0.0] for _ in space.words],
        )
        _, misses = score_category(broken, category, "3cosadd", 99, 1)
        for a, b, c, d, got in misses:
            self.assertIn(got, broken.words)
            self.assertNotEqual(got, d)

    def test_limit_caps_the_questions_asked(self):
        space, category = parallelogram()
        score, _ = score_category(space, category, "3cosadd", limit=3, seed=1)
        self.assertEqual(score.asked, 3)


class TestEvaluate(unittest.TestCase):
    def test_scores_every_method_on_every_category(self):
        words = []
        for stem in ["ship", "whale", "boat", "sail", "rope"]:
            words += [stem, stem + "s"]
        space = vocabulary(words)
        scores, misses = evaluate(space, limit=6)
        self.assertEqual({s.method for s in scores}, {"3cosadd", "3cosmul"})
        self.assertEqual({s.category for s in scores}, {"plural"})
        self.assertEqual(set(misses), {"3cosadd/plural", "3cosmul/plural"})

    def test_is_reproducible(self):
        space, _ = parallelogram()
        first, _ = evaluate(space, limit=6)
        second, _ = evaluate(space, limit=6)
        self.assertEqual(
            [(s.method, s.top1) for s in first],
            [(s.method, s.top1) for s in second],
        )


class TestTotals(unittest.TestCase):
    def test_sums_across_categories(self):
        scores = [
            Score("3cosadd", "plural", asked=10, top1=3, top5=5, form=7),
            Score("3cosadd", "past", asked=10, top1=1, top5=2, form=4),
            Score("3cosmul", "plural", asked=10, top1=2, top5=4, form=6),
        ]
        combined = totals(scores)
        self.assertEqual(combined["3cosadd"].asked, 20)
        self.assertEqual(combined["3cosadd"].top1, 4)
        self.assertEqual(combined["3cosadd"].form, 11)
        self.assertEqual(combined["3cosmul"].asked, 10)

    def test_labels_the_rollup(self):
        combined = totals([Score("m", "plural", 4, 1, 2, 3)])
        self.assertEqual(combined["m"].category, "ALL")

    def test_empty_input(self):
        self.assertEqual(totals([]), {})


if __name__ == "__main__":
    unittest.main()
