from __future__ import annotations

import math
import os
import tempfile
import unittest

from sens.space import METRICS, Space


def square() -> Space:
    """Four words at the corners of a unit square.

    Laid out so that `king -> queen` and `man -> woman` are the same offset,
    which makes the analogy answer knowable in advance rather than a matter
    of taste.
    """
    return Space(
        words=["king", "queen", "man", "woman"],
        vectors=[
            [1.0, 1.0],  # king
            [1.0, 2.0],  # queen   = king  + (0, 1)
            [3.0, 1.0],  # man
            [3.0, 2.0],  # woman   = man   + (0, 1)
        ],
    )


class TestBasics(unittest.TestCase):
    def test_length_and_membership(self):
        s = square()
        self.assertEqual(len(s), 4)
        self.assertIn("king", s)
        self.assertNotIn("jester", s)

    def test_dim(self):
        self.assertEqual(square().dim, 2)

    def test_vector_lookup(self):
        self.assertEqual(square().vector("man"), [3.0, 1.0])

    def test_unknown_word_raises_key_error(self):
        with self.assertRaises(KeyError):
            square().vector("jester")

    def test_direction_is_normalised(self):
        v = square().direction("king")
        self.assertAlmostEqual(math.sqrt(sum(x * x for x in v)), 1.0)


class TestSimilarity(unittest.TestCase):
    def test_a_word_is_identical_to_itself(self):
        self.assertAlmostEqual(square().similarity("king", "king"), 1.0)

    def test_similarity_is_symmetric(self):
        s = square()
        self.assertAlmostEqual(
            s.similarity("king", "woman"), s.similarity("woman", "king")
        )

    def test_orthogonal_vectors_score_zero(self):
        s = Space(words=["a", "b"], vectors=[[1.0, 0.0], [0.0, 1.0]])
        self.assertAlmostEqual(s.similarity("a", "b"), 0.0)

    def test_opposite_vectors_score_minus_one(self):
        s = Space(words=["a", "b"], vectors=[[1.0, 0.0], [-1.0, 0.0]])
        self.assertAlmostEqual(s.similarity("a", "b"), -1.0)

    def test_scale_does_not_matter(self):
        s = Space(words=["a", "b"], vectors=[[1.0, 1.0], [7.0, 7.0]])
        self.assertAlmostEqual(s.similarity("a", "b"), 1.0)


class TestNeighbors(unittest.TestCase):
    def test_excludes_the_query_word(self):
        self.assertNotIn("king", [w for w, _ in square().neighbors("king")])

    def test_respects_the_count(self):
        self.assertEqual(len(square().neighbors("king", n=2)), 2)

    def test_returns_descending_scores(self):
        scores = [s for _, s in square().neighbors("king")]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_finds_the_actually_nearest_direction(self):
        s = Space(
            words=["a", "near", "far"],
            vectors=[[1.0, 0.0], [1.0, 0.1], [-1.0, 0.0]],
        )
        self.assertEqual(s.neighbors("a", n=1)[0][0], "near")


class TestAnalogy(unittest.TestCase):
    def test_solves_a_parallelogram_it_was_built_to_solve(self):
        result = square().analogy("king", "queen", "man", n=1)
        self.assertEqual(result[0][0], "woman")

    def test_excludes_all_three_inputs(self):
        returned = {w for w, _ in square().analogy("king", "queen", "man")}
        self.assertFalse(returned & {"king", "queen", "man"})

    def test_runs_in_the_other_direction_too(self):
        result = square().analogy("queen", "king", "woman", n=1)
        self.assertEqual(result[0][0], "man")


def fan() -> Space:
    """Four words spread evenly around a quarter turn.

    They have to differ in angle, not just in length: everything in `Space`
    normalises first, so a set of collinear vectors is one point wearing
    four names and any ordering over it is meaningless.
    """
    h = math.sqrt(3) / 2
    return Space(
        words=["cold", "cool", "warm", "hot"],
        vectors=[[0.0, 1.0], [0.5, h], [h, 0.5], [1.0, 0.0]],
    )


class TestAnalogyMul(unittest.TestCase):
    def test_solves_the_parallelogram(self):
        self.assertEqual(
            square().analogy_mul("king", "queen", "man", n=1)[0][0], "woman"
        )

    def test_excludes_all_three_inputs(self):
        returned = {w for w, _ in square().analogy_mul("king", "queen", "man")}
        self.assertFalse(returned & {"king", "queen", "man"})

    def test_respects_the_count(self):
        self.assertEqual(len(square().analogy_mul("king", "queen", "man", n=1)), 1)

    def test_scores_are_descending(self):
        scores = [s for _, s in square().analogy_mul("king", "queen", "man")]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_scores_stay_positive(self):
        # Cosines are mapped into [0, 1] before multiplying, so no score can
        # come out negative however the vectors are arranged.
        s = Space(words=["a", "b", "c", "d"],
                  vectors=[[1.0, 0.0], [-1.0, 0.2], [0.0, -1.0], [0.5, 0.5]])
        for _, score in s.analogy_mul("a", "b", "c"):
            self.assertGreaterEqual(score, 0.0)

    def test_unknown_word_raises(self):
        with self.assertRaises(KeyError):
            square().analogy_mul("king", "queen", "jester")


class TestAxis(unittest.TestCase):
    def test_orders_from_the_negative_end_to_the_positive_end(self):
        s = fan()
        order = [w for w, _ in s.axis("cold", "hot", list(s.words))]
        self.assertEqual(order, ["cold", "cool", "warm", "hot"])

    def test_reversing_the_axis_reverses_the_order(self):
        s = fan()
        order = [w for w, _ in s.axis("hot", "cold", list(s.words))]
        self.assertEqual(order, ["hot", "warm", "cool", "cold"])

    def test_the_two_poles_land_on_opposite_signs(self):
        scores = dict(fan().axis("cold", "hot", ["cold", "hot"]))
        self.assertLess(scores["cold"], 0.0)
        self.assertGreater(scores["hot"], 0.0)

    def test_skips_words_outside_the_vocabulary(self):
        s = square()
        result = s.axis("king", "queen", ["man", "jester", "woman"])
        self.assertEqual({w for w, _ in result}, {"man", "woman"})

    def test_returns_a_score_per_word(self):
        result = square().axis("king", "queen", ["man", "woman"])
        self.assertEqual(len(result), 2)
        self.assertTrue(all(isinstance(s, float) for _, s in result))


class TestRank(unittest.TestCase):
    def test_ranks_an_arbitrary_query_vector(self):
        s = square()
        top = s.rank([1.0, 1.5], n=1)[0][0]
        self.assertIn(top, {"king", "queen"})

    def test_exclusions_are_honoured(self):
        s = square()
        returned = {w for w, _ in s.rank([1.0, 1.0], exclude={"king", "man"})}
        self.assertFalse(returned & {"king", "man"})


class TestPersistence(unittest.TestCase):
    def test_round_trips_through_a_file(self):
        original = Space(
            words=["alpha", "beta"],
            vectors=[[1.0, -2.0, 0.5], [0.25, 0.0, 3.0]],
            meta={"note": "hello"},
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "space.sens")
            original.save(path)
            restored = Space.load(path)

        self.assertEqual(restored.words, original.words)
        self.assertEqual(restored.meta, original.meta)
        self.assertEqual(restored.dim, 3)
        for a_row, b_row in zip(original.vectors, restored.vectors):
            for a, b in zip(a_row, b_row):
                self.assertAlmostEqual(a, b, places=6)

    def test_rejects_a_file_that_is_not_a_space(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "nope.sens")
            with open(path, "wb") as handle:
                handle.write(b"not a space at all")
            with self.assertRaises(ValueError):
                Space.load(path)

    def test_non_ascii_words_survive_the_round_trip(self):
        original = Space(words=["café", "naïve"], vectors=[[1.0], [2.0]])
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "space.sens")
            original.save(path)
            self.assertEqual(Space.load(path).words, ["café", "naïve"])


if __name__ == "__main__":
    unittest.main()


class TestMetrics(unittest.TestCase):
    """Cosine is a choice, so it has to be possible to choose otherwise."""

    def graded(self) -> Space:
        """Same directions, very different lengths.

        `long` points the same way as `query` but is ten times its length;
        `short` points the same way but is a tenth. Cosine cannot tell the
        three apart, and the other two metrics care about nothing else.
        """
        return Space(
            words=["query", "long", "short", "sideways"],
            vectors=[[1.0, 0.0], [10.0, 0.0], [0.1, 0.0], [0.0, 1.0]],
        )

    def test_cosine_ignores_length(self):
        space = self.graded()
        self.assertAlmostEqual(
            space.similarity("query", "long", metric="cosine"), 1.0
        )
        self.assertAlmostEqual(
            space.similarity("query", "short", metric="cosine"), 1.0
        )

    def test_dot_is_dominated_by_length(self):
        space = self.graded()
        self.assertEqual(space.neighbors("query", n=1, metric="dot")[0][0],
                         "long")

    def test_euclidean_prefers_the_nearest_point(self):
        space = self.graded()
        self.assertEqual(
            space.neighbors("query", n=1, metric="euclidean")[0][0], "short"
        )

    def test_all_three_agree_on_a_unit_length_space(self):
        # With every vector already normalised there is nothing for the
        # metrics to disagree about, which is a useful sanity anchor.
        h = math.sqrt(3) / 2
        space = Space(
            words=["a", "b", "c"],
            vectors=[[1.0, 0.0], [h, 0.5], [0.0, 1.0]],
        )
        picks = {
            m: space.neighbors("a", n=1, metric=m)[0][0]
            for m in METRICS
        }
        self.assertEqual(set(picks.values()), {"b"})

    def test_an_unknown_metric_is_rejected(self):
        for call in (
            lambda: self.graded().similarity("query", "long", metric="taxicab"),
            lambda: self.graded().neighbors("query", metric="taxicab"),
        ):
            with self.assertRaises(ValueError) as caught:
                call()
            self.assertIn("taxicab", str(caught.exception))

    def test_analogy_builds_its_target_in_the_metric_own_space(self):
        # Regression test. The target used to be assembled from unit vectors
        # whatever the metric was, so a length-sensitive metric compared a
        # target of magnitude ~1 against candidates of magnitude ~100; every
        # candidate was about equally far away and the shortest vector always
        # won. Euclidean analogy scored a clean 0% until this was fixed.
        #
        # Here the parallelogram is exact but the vectors are long, so a
        # target built in unit space would miss `woman` entirely.
        space = Space(
            words=["king", "queen", "man", "woman", "tiny"],
            vectors=[
                [100.0, 100.0], [100.0, 200.0],
                [300.0, 100.0], [300.0, 200.0],
                [0.01, 0.01],
            ],
        )
        for metric in ("dot", "euclidean"):
            self.assertEqual(
                space.analogy("king", "queen", "man", n=1, metric=metric)[0][0],
                "woman",
                metric,
            )

    def test_analogy_still_excludes_its_inputs_under_every_metric(self):
        space = self.graded()
        for metric in METRICS:
            returned = {
                w for w, _ in
                space.analogy("query", "long", "short", metric=metric)
            }
            self.assertFalse(returned & {"query", "long", "short"}, metric)

    def test_metrics_tuple_lists_what_is_accepted(self):
        space = self.graded()
        for metric in METRICS:
            space.neighbors("query", n=1, metric=metric)
