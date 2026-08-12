"""The README is a deliverable, so it gets checked like one.

Two things here rot silently. Internal links break when a heading is
reworded, and there is no compiler to notice. And the summary table is only
worth having if it points at sections that exist.
"""

from __future__ import annotations

import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = os.path.join(ROOT, "README.md")


def slug(heading: str) -> str:
    """GitHub's anchor rule: lowercase, drop punctuation, hyphenate spaces."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text)


class TestReadmeLinks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(README, encoding="utf-8") as handle:
            cls.text = handle.read()
        cls.headings = {
            slug(line.lstrip("#"))
            for line in cls.text.splitlines()
            if line.startswith("#")
        }
        # Markdown links whose target is an in-page anchor.
        cls.anchors = re.findall(r"\]\(#([^)]+)\)", cls.text)

    def test_there_are_internal_links_to_check(self):
        self.assertGreater(len(self.anchors), 10)

    def test_every_internal_link_resolves(self):
        missing = sorted({a for a in self.anchors if a not in self.headings})
        self.assertEqual(missing, [], f"dangling anchors: {missing}")

    def test_headings_are_unique(self):
        seen = [
            slug(line.lstrip("#"))
            for line in self.text.splitlines()
            if line.startswith("#")
        ]
        duplicates = sorted({s for s in seen if seen.count(s) > 1})
        self.assertEqual(duplicates, [], f"duplicate headings: {duplicates}")

    def test_no_relative_file_link_is_broken(self):
        targets = re.findall(r"\]\((?!#)(?!https?:)([^)]+)\)", self.text)
        for target in targets:
            path = os.path.join(ROOT, target.split("#")[0])
            self.assertTrue(os.path.exists(path), target)


class TestReadmeClaims(unittest.TestCase):
    """A few figures that must not drift out of step with the code."""

    @classmethod
    def setUpClass(cls):
        with open(README, encoding="utf-8") as handle:
            cls.text = handle.read()

    def test_the_test_count_is_current(self):
        # Counted by reading the files, which sounds crude and is the only
        # version that terminates. Shelling out to `unittest discover` makes
        # the suite rediscover this file and shell out again; that hung
        # until it was killed. Calling `discover` in-process hangs too, being
        # re-entrant on the loader already running. A static count is
        # neither, and agrees exactly with `discover` run standalone because
        # every test here is a plain method.
        #
        # `[ \t]` rather than `\s`: under re.MULTILINE, `\s` also matches the
        # newline, so `^\s+` spans blank lines and backtracks badly.
        total = 0
        for name in sorted(os.listdir(os.path.join(ROOT, "tests"))):
            if not name.startswith("test_") or not name.endswith(".py"):
                continue
            path = os.path.join(ROOT, "tests", name)
            with open(path, encoding="utf-8") as handle:
                total += len(
                    re.findall(r"^[ \t]+def test_", handle.read(), re.M)
                )
        claimed = re.findall(r"(\d+) tests", self.text)
        self.assertIn(str(total), claimed,
                      f"README claims {claimed}, suite has {total}")

    def test_every_cli_command_in_the_readme_exists(self):
        from sens.__main__ import main

        named = set(re.findall(r"python -m sens (\w[\w-]*)", self.text))
        for command in sorted(named):
            if command in {"fetch"}:
                continue  # network
            with self.assertRaises(SystemExit) as caught:
                main([command, "--help"])
            self.assertEqual(caught.exception.code, 0, command)


if __name__ == "__main__":
    unittest.main()
