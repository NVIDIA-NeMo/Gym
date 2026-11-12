# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard benchmark correctness: unknown changes, XML parity, and failed speed claims."""

import tempfile
import unittest
from pathlib import Path

from compare import is_pilot_path, junit, same_outcomes, select_targets, selected_outcomes, summarize


class ComparisonTest(unittest.TestCase):
    def test_yaml_change_selects_explicit_dependents(self):
        graph = {"test:a": {"a.py", "config.yaml"}, "test:b": {"b.py"}}
        self.assertEqual(select_targets({"config.yaml"}, graph), (["test:a"], []))

    def test_unknown_input_forces_all_tests(self):
        graph = {"test:a": {"a.py"}, "test:b": {"b.py"}}
        self.assertEqual(select_targets({"undeclared.yaml"}, graph), (["test:a", "test:b"], ["undeclared.yaml"]))

    def test_unchanged_files_select_no_tests(self):
        self.assertEqual(select_targets(set(), {"test:a": {"a.py"}}), ([], []))

    def test_junit_counts_cases_not_nested_suite_totals(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "results.xml").write_text(
                '<testsuites tests="2"><testsuite tests="2">'
                '<testcase classname="tests.foo" name="test_ok"/>'
                '<testcase classname="tests.foo" name="test_skip"><skipped/></testcase>'
                "</testsuite></testsuites>"
            )
            report = junit(directory)
            self.assertEqual(report["count"], 2)
            self.assertEqual(report["cases"][1][2], "skipped")
            self.assertTrue(same_outcomes(report, report))
            changed = {**report, "cases": [["tests.foo", "test_ok", "passed"], ["tests.foo", "test_skip", "passed"]]}
            self.assertFalse(same_outcomes(report, changed))
            self.assertFalse(same_outcomes({"count": 0, "cases": []}, {"count": 0, "cases": []}))

    def test_selected_parity_keeps_every_case_in_selected_file(self):
        full = {
            "cases": [
                ["tests.test_a.TestThing", "test_first", "passed"],
                ["tests.test_a.TestThing", "test_second", "passed"],
                ["tests.test_b", "test_other", "passed"],
            ]
        }
        expected = selected_outcomes(full, ["tests/test_a.py:tests"])
        self.assertEqual(len(expected), 2)
        self.assertNotEqual(expected, [full["cases"][0]])
        self.assertEqual(selected_outcomes(full, []), [])
        with self.assertRaisesRegex(ValueError, "Cannot map"):
            selected_outcomes(full, ["tests/test_missing.py:tests"])

    def test_overlay_cannot_hide_source_changes(self):
        self.assertTrue(is_pilot_path("tests/BUILD"))
        self.assertTrue(is_pilot_path("scripts/pants/pilot.lock"))
        self.assertFalse(is_pilot_path("nemo_gym/utils/text.py"))
        self.assertFalse(is_pilot_path("pyproject.toml"))

    def test_failed_comparison_never_claims_speedup(self):
        report = {
            "base_sha": "a",
            "parent_sha": "b",
            "pilot_sha": "c",
            "overlay_sha256": "d",
            "measurements": [{"case": "cold", "uv": {"seconds": 10}, "pants": {"seconds": 1}, "valid": False}],
        }
        summary = summarize(report)
        self.assertIn("withheld", summary)
        self.assertNotIn("10.00x", summary)


if __name__ == "__main__":
    unittest.main()
