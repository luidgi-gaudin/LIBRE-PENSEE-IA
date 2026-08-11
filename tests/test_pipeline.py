"""End-to-end tests on a tiny synthetic corpus.

The corpus below is built so the right answer is a matter of arithmetic
rather than of literary judgement: two families of words that never share a
sentence, and two words that are perfect substitutes for each other. If the
pipeline cannot separate those, nothing it says about Melville is worth
reading either.
"""

from __future__ import annotations

import os
import tempfile
import unittest

from sens.pipeline import Config, build

SEA = "the sailor sails the ship across the wide salt sea "
OCEAN = "the sailor sails the ship across the wide salt ocean "
FOREST = "a hunter walks a path through a dark green forest "
GROVE = "a hunter walks a path through a dark green grove "

# Repetition is doing the work of corpus size here: PPMI needs enough
# observations that a co-occurrence stops looking like an accident.
SYNTHETIC = (SEA + FOREST + OCEAN + GROVE) * 200


class TestBuild(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = os.path.join(cls.tmp.name, "synthetic.txt")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(SYNTHETIC)
        cls.path = path
        cls.config = Config(
            vocab_size=50,
            min_count=1,
            window=4,
            dim=8,
            oversample=8,
            min_pair_weight=0.0,
        )
        cls.space, cls.report = build([path], cls.config, verbose=False)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_vocabulary_covers_the_corpus(self):
        for word in ["sailor", "ship", "sea", "ocean", "hunter", "forest"]:
            self.assertIn(word, self.space)

    def test_dimensionality_matches_the_config(self):
        self.assertEqual(self.space.dim, 8)

    def test_interchangeable_words_end_up_close(self):
        # `sea` and `ocean` never co-occur; they occupy the same slot in
        # otherwise identical sentences. Distributional similarity is the
        # only thing that could possibly relate them, so this is the test
        # that the whole idea works.
        self.assertGreater(self.space.similarity("sea", "ocean"), 0.9)
        self.assertGreater(self.space.similarity("forest", "grove"), 0.9)

    def test_words_from_different_contexts_stay_apart(self):
        across = self.space.similarity("sea", "forest")
        within = self.space.similarity("sea", "ocean")
        self.assertLess(across, within)

    def test_nearest_neighbour_of_sea_is_ocean(self):
        self.assertEqual(self.space.neighbors("sea", n=1)[0][0], "ocean")

    def test_nearest_neighbour_of_hunter_comes_from_its_own_world(self):
        neighbours = {w for w, _ in self.space.neighbors("hunter", n=4)}
        self.assertTrue(neighbours & {"path", "walks", "forest", "grove"})

    def test_analogy_crosses_between_the_two_worlds(self):
        # sailor : sea :: hunter : ?  should land in the forest half.
        answers = {
            w for w, _ in self.space.analogy("sailor", "sea", "hunter", n=5)
        }
        self.assertTrue(answers & {"forest", "grove", "path"})

    def test_report_covers_every_stage(self):
        names = [name for name, _, _ in self.report.stages]
        self.assertEqual(
            names,
            ["read", "vocabulary", "co-occurrence", "ppmi", "factorise"],
        )

    def test_report_renders(self):
        self.assertIn("ppmi", self.report.render())

    def test_metadata_records_how_it_was_built(self):
        self.assertEqual(self.space.meta["config"]["dim"], 8)
        self.assertEqual(self.space.meta["sources"], [self.path])
        self.assertGreater(self.space.meta["tokens"], 0)

    def test_the_build_is_reproducible(self):
        again, _ = build([self.path], self.config, verbose=False)
        self.assertEqual(again.words, self.space.words)
        for a_row, b_row in zip(again.vectors, self.space.vectors):
            for a, b in zip(a_row, b_row):
                self.assertAlmostEqual(a, b, places=9)

    def test_a_different_seed_gives_a_different_basis_but_same_geometry(self):
        # The random sketch fixes the coordinate frame, not the distances.
        # Cosine similarities should survive re-seeding.
        other = Config(**{**self.config.as_dict(), "seed": 7})
        shifted, _ = build([self.path], other, verbose=False)
        self.assertAlmostEqual(
            shifted.similarity("sea", "ocean"),
            self.space.similarity("sea", "ocean"),
            places=4,
        )


class TestConfig(unittest.TestCase):
    def test_as_dict_round_trips_into_a_config(self):
        original = Config(dim=32, window=7)
        restored = Config(**original.as_dict())
        self.assertEqual(restored.dim, 32)
        self.assertEqual(restored.window, 7)


if __name__ == "__main__":
    unittest.main()
