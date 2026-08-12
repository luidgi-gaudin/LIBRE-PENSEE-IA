from __future__ import annotations

import unittest

from sens.text import (
    Vocabulary,
    normalise,
    strip_boilerplate,
    tokenize,
)


class TestNormalise(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(normalise("Call Me ISHMAEL"), "call me ishmael")

    def test_folds_curly_apostrophes(self):
        self.assertEqual(normalise("don’t"), "don't")

    def test_strips_accents(self):
        self.assertEqual(normalise("naïve café"), "naive cafe")


class TestTokenize(unittest.TestCase):
    def test_splits_on_punctuation(self):
        self.assertEqual(
            list(tokenize("Call me Ishmael. Some years ago—")),
            ["call", "me", "ishmael", "some", "years", "ago"],
        )

    def test_keeps_internal_apostrophes(self):
        self.assertEqual(list(tokenize("the whale's o'clock")),
                         ["the", "whale's", "o'clock"])

    def test_drops_digits(self):
        self.assertEqual(list(tokenize("chapter 42 loomings")),
                         ["chapter", "loomings"])

    def test_leading_apostrophe_is_not_part_of_the_word(self):
        self.assertEqual(list(tokenize("'tis")), ["tis"])

    def test_empty_input_yields_nothing(self):
        self.assertEqual(list(tokenize("")), [])


class TestBoilerplate(unittest.TestCase):
    def test_removes_header_and_footer(self):
        text = (
            "legal preamble\n"
            "*** START OF THE PROJECT GUTENBERG EBOOK MOBY-DICK ***\n"
            "the actual book\n"
            "*** END OF THE PROJECT GUTENBERG EBOOK MOBY-DICK ***\n"
            "licence terms\n"
        )
        self.assertEqual(strip_boilerplate(text).strip(), "the actual book")

    def test_handles_the_older_this_wording(self):
        text = (
            "*** START OF THIS PROJECT GUTENBERG EBOOK X ***\n"
            "body\n"
            "*** END OF THIS PROJECT GUTENBERG EBOOK X ***\n"
        )
        self.assertEqual(strip_boilerplate(text).strip(), "body")

    def test_plain_text_passes_through_untouched(self):
        self.assertEqual(strip_boilerplate("just a book"), "just a book")


class TestVocabulary(unittest.TestCase):
    def setUp(self):
        self.tokens = (
            ["the"] * 10 + ["whale"] * 6 + ["sea"] * 6 + ["rare"] * 1
        )

    def test_orders_by_descending_frequency(self):
        vocab = Vocabulary.from_tokens(self.tokens, min_count=1)
        self.assertEqual(vocab.words[0], "the")

    def test_breaks_frequency_ties_alphabetically(self):
        vocab = Vocabulary.from_tokens(self.tokens, min_count=1)
        self.assertEqual(vocab.words[1:3], ["sea", "whale"])

    def test_min_count_drops_the_tail(self):
        vocab = Vocabulary.from_tokens(self.tokens, min_count=2)
        self.assertNotIn("rare", vocab)

    def test_max_size_truncates_from_the_rare_end(self):
        vocab = Vocabulary.from_tokens(self.tokens, min_count=1, max_size=2)
        self.assertEqual(vocab.words, ["the", "sea"])

    def test_a_smaller_vocabulary_is_a_prefix_of_a_larger_one(self):
        big = Vocabulary.from_tokens(self.tokens, min_count=1, max_size=4)
        small = Vocabulary.from_tokens(self.tokens, min_count=1, max_size=2)
        self.assertEqual(big.words[: len(small)], small.words)

    def test_raising_min_count_also_leaves_a_prefix(self):
        # Load-bearing for audit_vocabulary, which scores several
        # vocabularies against one truth table by assuming that index i is
        # the same word in all of them. Both cuts remove a suffix of the
        # frequency-sorted list, so every vocabulary from a given corpus is
        # a prefix of every larger one.
        loose = Vocabulary.from_tokens(self.tokens, min_count=1)
        strict = Vocabulary.from_tokens(self.tokens, min_count=6)
        self.assertEqual(loose.words[: len(strict)], strict.words)

    def test_the_two_cuts_agree_with_each_other(self):
        # Reaching a given size by raising min_count or by lowering max_size
        # must give the same words, or "the vocabularies are nested" would
        # depend on which knob you turned.
        by_count = Vocabulary.from_tokens(self.tokens, min_count=6)
        by_size = Vocabulary.from_tokens(
            self.tokens, min_count=1, max_size=len(by_count)
        )
        self.assertEqual(by_count.words, by_size.words)

    def test_counts_line_up_with_words(self):
        vocab = Vocabulary.from_tokens(self.tokens, min_count=1)
        self.assertEqual(vocab.counts[vocab.index["whale"]], 6)

    def test_total_is_the_sum_of_kept_counts(self):
        vocab = Vocabulary.from_tokens(self.tokens, min_count=2)
        self.assertEqual(vocab.total, 22)

    def test_encode_maps_to_indices(self):
        vocab = Vocabulary.from_tokens(self.tokens, min_count=1)
        self.assertEqual(
            vocab.encode(["the", "whale"]),
            [vocab.index["the"], vocab.index["whale"]],
        )

    def test_encode_drops_unknown_words(self):
        vocab = Vocabulary.from_tokens(self.tokens, min_count=2)
        self.assertEqual(len(vocab.encode(["the", "rare", "sea"])), 2)

    def test_membership_and_length(self):
        vocab = Vocabulary.from_tokens(self.tokens, min_count=2)
        self.assertIn("sea", vocab)
        self.assertNotIn("kraken", vocab)
        self.assertEqual(len(vocab), 3)


if __name__ == "__main__":
    unittest.main()


class TestTokenisationOptions(unittest.TestCase):
    """The choices made before any counting happens."""

    def test_case_is_folded_by_default(self):
        self.assertEqual(list(tokenize("The Whale")), ["the", "whale"])

    def test_case_can_be_kept(self):
        self.assertEqual(
            list(tokenize("The Whale", lowercase=False)), ["The", "Whale"]
        )

    def test_keeping_case_splits_a_word_from_itself(self):
        # The cost of not folding: two vocabulary entries for one word, each
        # built from half the evidence.
        kept = list(tokenize("The whale. The whale.", lowercase=False))
        self.assertEqual(kept.count("The"), 2)
        self.assertEqual(kept.count("whale"), 2)
        self.assertNotIn("the", kept)

    def test_accents_are_folded_by_default(self):
        self.assertEqual(list(tokenize("Mercédès")), ["mercedes"])

    def test_not_folding_accents_shatters_the_word(self):
        # The token pattern is ASCII letters, so an unfolded accented
        # character is a boundary rather than a letter. This is why the
        # option is a real fork and not a cosmetic one.
        self.assertEqual(
            list(tokenize("Mercédès", fold_accents=False)), ["merc", "d", "s"]
        )

    def test_clitics_stay_attached_by_default(self):
        self.assertEqual(list(tokenize("the whale's jaw")),
                         ["the", "whale's", "jaw"])

    def test_clitics_can_be_split_off(self):
        self.assertEqual(
            list(tokenize("the whale's jaw", split_clitics=True)),
            ["the", "whale", "'s", "jaw"],
        )

    def test_splitting_handles_contractions_too(self):
        self.assertEqual(
            list(tokenize("don't", split_clitics=True)), ["don", "'t"]
        )

    def test_splitting_leaves_plain_words_alone(self):
        self.assertEqual(
            list(tokenize("the whale swims", split_clitics=True)),
            ["the", "whale", "swims"],
        )

    def test_splitting_never_yields_an_empty_stem(self):
        # `'tis` loses its leading apostrophe to the token pattern, so there
        # is nothing here to split; the guard is against emitting "".
        for token in tokenize("'tis o'clock", split_clitics=True):
            self.assertTrue(token)

    def test_a_shared_clitic_becomes_one_token_for_every_word(self):
        # The substantive consequence: possession stops being a property of
        # each noun and becomes a word in its own right.
        tokens = list(tokenize("the whale's and the ship's", split_clitics=True))
        self.assertEqual(tokens.count("'s"), 2)

    def test_normalise_respects_both_flags(self):
        self.assertEqual(normalise("Café", lowercase=False), "Cafe")
        self.assertEqual(normalise("Café", fold_accents=False), "café")

    def test_curly_apostrophes_are_folded_regardless(self):
        for kwargs in ({}, {"lowercase": False}, {"fold_accents": False}):
            self.assertIn("'", normalise("don’t", **kwargs))
