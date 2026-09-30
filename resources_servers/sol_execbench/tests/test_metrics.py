# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fixed-denominator and unknown-result regression tests for SOL aggregation."""

import unittest
from copy import deepcopy

from resources_servers.sol_execbench.metrics import aggregate_sol_results


PROTOCOL = "a" * 64


def result(task_id: str, index: int, score: float = 0.0, *, solved: bool = False, timeout: bool = False) -> dict:
    return {
        "task_id": task_id,
        "_ng_rollout_index": index,
        "protocol_sha256": PROTOCOL,
        "outcome": "EVALUATION_TIMEOUT" if timeout else "PASSED" if solved else "CANDIDATE_FAILED",
        "infrastructure_error": timeout,
        "solved": solved,
        "sol_score": None if timeout else score,
        "mask_sample": timeout,
    }


class TestSolMetrics(unittest.TestCase):
    def aggregate(self, rows: list[dict], **kwargs):
        params = {"task_ids": ["task-a", "task-b"], "samples_per_task": 2, "protocol_sha256": PROTOCOL}
        params.update(kwargs)
        return aggregate_sol_results(rows, **params)

    def test_best_passing_score_and_fixed_denominators(self):
        rows = [
            result("task-a", 0, 0.2, solved=True),
            result("task-a", 1, 0.8, solved=True),
            result("task-b", 0),
            result("task-b", 1),
        ]
        before = deepcopy(rows)
        metrics = self.aggregate(rows)
        self.assertEqual(
            metrics.key_metrics,
            {"official/sol_best2": 0.4, "official/correctness_at_1": 0.5, "official/pass_at_2": 0.5},
        )
        self.assertEqual(metrics.agent_metrics["counts/candidate_failures"], 2)
        self.assertEqual(metrics.agent_metrics["outcomes/PASSED"], 2)
        self.assertEqual(metrics.group_level_metrics[1]["observed_best_passing_sol"], 0.0)
        self.assertEqual(rows, before)
        self.assertFalse(any("reward" in key or "mean/" in key for key in metrics.agent_metrics))
        self.assertFalse(any(key.startswith("timeout_zero/") for key in metrics.agent_metrics))
        for key, value in metrics.key_metrics.items():
            self.assertEqual(metrics.agent_metrics[key], value)

    def test_headline_zero_floor_preserves_raw_passing_scores(self):
        rows = [result("task-a", 0, 1.7, solved=True), result("task-b", 0, -0.2, solved=True)]
        before = deepcopy(rows)
        metrics = self.aggregate(rows, samples_per_task=1, timeout_zero_sensitivity=True)
        self.assertEqual(metrics.key_metrics["official/sol_best1"], 0.85)
        self.assertEqual(metrics.key_metrics["timeout_zero/sol_best1"], 0.85)
        self.assertEqual(metrics.key_metrics["official/pass_at_1"], 1.0)
        self.assertEqual(metrics.group_level_metrics[1]["observed_best_passing_sol"], -0.2)
        self.assertEqual(rows, before)

    def test_all_failures_are_complete_zero(self):
        metrics = self.aggregate([result(t, i) for t in ["task-a", "task-b"] for i in range(2)])
        self.assertTrue(metrics.agent_metrics["complete"])
        self.assertEqual(set(metrics.key_metrics.values()), {0.0})

    def test_timeout_sensitivity_does_not_change_official(self):
        rows = [
            result("task-a", 0, 0.8, solved=True),
            result("task-a", 1, timeout=True),
            result("task-b", 0),
            result("task-b", 1),
        ]
        metrics = self.aggregate(rows, timeout_zero_sensitivity=True)
        self.assertTrue(
            all(value is None for key, value in metrics.key_metrics.items() if key.startswith("official/"))
        )
        self.assertEqual(metrics.key_metrics["timeout_zero/sol_best2"], 0.4)
        self.assertEqual(metrics.key_metrics["timeout_zero/correctness_at_1"], 0.25)
        self.assertEqual(metrics.key_metrics["timeout_zero/pass_at_2"], 0.5)
        self.assertTrue(metrics.agent_metrics["timeout_zero/eligible"])
        self.assertEqual(metrics.agent_metrics["counts/unresolved_samples"], 1)
        self.assertEqual(metrics.agent_metrics["counts/infrastructure_errors"], 1)
        self.assertEqual(metrics.agent_metrics["counts/missing_samples"], 0)

    def test_missing_entire_task_is_not_dropped(self):
        metrics = self.aggregate(
            [result("task-a", 0, 0.8, solved=True), result("task-a", 1)], timeout_zero_sensitivity=True
        )
        self.assertTrue(all(value is None for value in metrics.key_metrics.values()))
        self.assertFalse(metrics.agent_metrics["timeout_zero/eligible"])
        self.assertEqual(metrics.agent_metrics["counts/expected_samples"], 4)
        self.assertEqual(metrics.agent_metrics["counts/observed_samples"], 2)
        self.assertEqual(metrics.agent_metrics["counts/missing_samples"], 2)
        self.assertEqual(metrics.group_level_metrics[1]["task_id"], "task-b")
        self.assertEqual(metrics.group_level_metrics[1]["missing_samples"], 2)
        self.assertEqual(metrics.group_level_metrics[1]["outcome_counts"], {})

    def test_all_missing_still_reports_expected_manifest(self):
        metrics = self.aggregate([], timeout_zero_sensitivity=True)
        self.assertEqual(metrics.agent_metrics["counts/unresolved_samples"], 4)
        self.assertEqual(metrics.agent_metrics["counts/observed_passes"], 0)
        self.assertEqual(len(metrics.group_level_metrics), 2)
        self.assertTrue(all(value is None for value in metrics.key_metrics.values()))

    def test_other_infrastructure_suppresses_sensitivity(self):
        rows = [result(t, i, timeout=True) for t in ["task-a", "task-b"] for i in range(2)]
        metrics = self.aggregate(rows, timeout_zero_sensitivity=True)
        self.assertEqual(metrics.key_metrics["timeout_zero/sol_best2"], 0.0)
        rows[0]["outcome"] = "ENVIRONMENT_FAILURE"
        metrics = self.aggregate(rows, timeout_zero_sensitivity=True)
        self.assertTrue(all(value is None for value in metrics.key_metrics.values()))
        self.assertFalse(metrics.agent_metrics["timeout_zero/eligible"])

    def test_duplicate_and_unexpected_identities(self):
        row = result("task-a", 0)
        with self.assertRaises(ValueError):
            self.aggregate([row, dict(row)])
        for field, values in {
            "task_id": ["unknown", None, [], 0],
            "_ng_rollout_index": [-1, 2, None, "0", 0.0, True],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                    self.aggregate([{**row, field: value}])

    def test_protocol_and_required_fields(self):
        row = result("task-a", 0)
        with self.assertRaises(ValueError):
            self.aggregate([{**row, "protocol_sha256": "b" * 64}])
        for field in [
            "task_id",
            "_ng_rollout_index",
            "protocol_sha256",
            "outcome",
            "infrastructure_error",
            "solved",
            "sol_score",
        ]:
            bad = dict(row)
            bad.pop(field)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.aggregate([bad])

    def test_invalid_scores_and_flags(self):
        for score in [True, False, float("nan"), float("inf"), -float("inf"), "0.5", None]:
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.aggregate([result("task-a", 0, score, solved=True)])
        for changes in [
            {"sol_score": 0.1},
            {"solved": 1},
            {"infrastructure_error": 0},
            {"mask_sample": True},
            {"mask_sample": 0},
            {"outcome": ""},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.aggregate([{**result("task-a", 0), **changes}])
        for changes in [{"solved": True}, {"sol_score": 0.0}, {"mask_sample": False}]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.aggregate([{**result("task-a", 0, timeout=True), **changes}])

    def test_mask_is_optional_but_raw_value_must_be_boolean(self):
        row = result("task-a", 0)
        row.pop("mask_sample")
        self.assertEqual(self.aggregate([row]).agent_metrics["counts/observed_samples"], 1)
        for value in [None, "false", 0, 1]:
            with self.subTest(mask=value), self.assertRaises(ValueError):
                self.aggregate([{**row, "mask_sample": value}])

    def test_outcome_and_solved_flags_must_agree(self):
        corrupted = [
            {**result("task-a", 0), "outcome": "PASSED"},
            {**result("task-a", 0, 0.8, solved=True), "outcome": "CANDIDATE_FAILED"},
            {**result("task-a", 0, 0.8, solved=True), "outcome": "NEW_FAILURE"},
            {**result("task-a", 0, timeout=True), "outcome": "PASSED"},
        ]
        for row in corrupted:
            with self.subTest(row=row), self.assertRaises(ValueError):
                self.aggregate([row])

    def test_only_measured_outcomes_can_be_scored(self):
        for outcome in [
            "CANDIDATE_FAILED",
            "CANDIDATE_SYNTAX_ERROR",
            "COMPILE_ERROR",
            "NO_SOLUTION",
            "INVALID_SOLUTION",
            "CANDIDATE_IMPORT_ERROR",
            "MISSING_ENTRYPOINT",
        ]:
            with self.subTest(outcome=outcome):
                row = {**result("task-a", 0), "outcome": outcome}
                metrics = self.aggregate([row], task_ids=["task-a"], samples_per_task=1)
                self.assertTrue(metrics.agent_metrics["complete"])
                self.assertEqual(metrics.key_metrics["official/sol_best1"], 0.0)
        for outcome in ["EVALUATION_TIMEOUT", "ENVIRONMENT_FAILURE", "NOT_EVALUATED", "NEW_FAILURE"]:
            with self.subTest(outcome=outcome), self.assertRaises(ValueError):
                self.aggregate([{**result("task-a", 0), "outcome": outcome}])

    def test_unknown_outcome_remains_unresolved(self):
        row = {**result("task-a", 0, timeout=True), "outcome": "NEW_FAILURE"}
        metrics = self.aggregate([row], task_ids=["task-a"], samples_per_task=1, timeout_zero_sensitivity=True)
        self.assertEqual(metrics.agent_metrics["counts/infrastructure_errors"], 1)
        self.assertEqual(metrics.agent_metrics["outcomes/NEW_FAILURE"], 1)
        self.assertTrue(all(value is None for value in metrics.key_metrics.values()))

    def test_invalid_manifest(self):
        for changes in [
            {"task_ids": []},
            {"task_ids": ["task-a", "task-a"]},
            {"task_ids": [""]},
            {"task_ids": [1]},
            {"samples_per_task": 0},
            {"samples_per_task": True},
            {"samples_per_task": 1.5},
            {"protocol_sha256": ""},
            {"timeout_zero_sensitivity": 1},
        ]:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.aggregate([], **changes)

    def test_order_does_not_change_results(self):
        rows = [result(t, i, 0.6, solved=True) for t in ["task-a", "task-b"] for i in range(2)]
        self.assertEqual(self.aggregate(rows), self.aggregate(list(reversed(rows))))


if __name__ == "__main__":
    unittest.main()
