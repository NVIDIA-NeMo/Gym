# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression checks for recovery and score integrity without GPU allocation."""

import json
import tempfile
import unittest
from pathlib import Path

from evaluate import audit_likelihood_parity, read_results


class ParityTests(unittest.TestCase):
    def test_bf16_near_tie_records_argmax_disagreement(self):
        report = audit_likelihood_parity([[-2.0, -2.001, -3.0, -4.0]], [[-2.002, -2.0, -3.0, -4.0]])
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["numerical_near_ties"]), 1)
        self.assertAlmostEqual(report["max_logprob_difference"], 0.002)

    def test_large_likelihood_discrepancy_still_fails(self):
        with self.assertRaises(AssertionError):
            audit_likelihood_parity([[-1.0, -2.0, -3.0, -4.0]], [[-2.0, -1.0, -3.0, -4.0]])


class ResumeTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "scores.jsonl"
        self.expected = {"hi/one": {"answer": "B"}}
        self.row = dict(
            id="hi/one", identity="current", choice_logprobs=[-4, -1, -3, -5], prediction="B", correct=True
        )

    def test_resume_preserves_completed_rows_and_repairs_partial_tail(self):
        complete = json.dumps(self.row) + "\n"
        self.path.write_text(complete + '{"id":')
        rows = read_results(self.path, self.expected, "current")
        self.assertEqual(list(rows), ["hi/one"])
        self.assertEqual(self.path.read_text(), complete)

    def test_rejects_stale_results(self):
        self.path.write_text(json.dumps(self.row) + "\n")
        with self.assertRaises(AssertionError):
            read_results(self.path, self.expected, "different-data-or-model")

    def test_rejects_misgraded_answer(self):
        self.row["correct"] = False
        self.path.write_text(json.dumps(self.row) + "\n")
        with self.assertRaises(AssertionError):
            read_results(self.path, self.expected, "current")

    def test_rejects_duplicate_results(self):
        self.path.write_text((json.dumps(self.row) + "\n") * 2)
        with self.assertRaises(AssertionError):
            read_results(self.path, self.expected, "current")

    def test_rejects_nonfinite_scores(self):
        self.row["choice_logprobs"][0] = float("nan")
        self.path.write_text(json.dumps(self.row) + "\n")
        with self.assertRaises(AssertionError):
            read_results(self.path, self.expected, "current")


if __name__ == "__main__":
    unittest.main()
