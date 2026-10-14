# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests of the validation summary.

The report gym dataset validate prints from its rollout and failure files, its keys and its
verdict.
"""

import json

import pytest


class TestValidationSummary:
    def test_summarizes_rollouts(self, tmp_path):
        from nemo_gym.tasks.harbor.validation_report import summarize_validation

        # Rows are the verify response as returned plus the collector's `_ng_task_id` stamp.
        rows = [
            {
                "_ng_task_id": {"taskset": "ds", "task_id": "solved"},
                "reward": 1.0,
                "mask_sample": False,
                "response": {"metadata": {"oracle": "solved"}},
            },
            {
                "_ng_task_id": {"taskset": "ds", "task_id": "wrong"},
                "reward": 0.0,
                "mask_sample": False,
                "failure_kind": "harbor:missing_reward",
                "response": {"metadata": {"oracle": "solved"}},
            },
            {
                "_ng_task_id": {"taskset": "ds", "task_id": "skipped"},
                "reward": 0.0,
                "mask_sample": False,
                "response": {"metadata": {"oracle": "unvalidated"}},
            },
            {
                "_ng_task_id": {"taskset": "ds", "task_id": "masked"},
                "reward": 0.0,
                "mask_sample": True,
                "failure_kind": "provider_unavailable",
            },
        ]
        rollouts = tmp_path / "rollouts.jsonl"
        rollouts.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        # An episode that failed before verification lands in the failures sidecar.
        (tmp_path / "rollouts_failures.jsonl").write_text(
            json.dumps(
                {
                    "_ng_task_id": {"taskset": "ds", "task_id": "broken"},
                    "_ng_failure_class": "environment_server_failed",
                    "_ng_failure_terminal": True,
                    "_ng_failure_message": "seed exploded",
                }
            )
            + "\n"
        )

        report = summarize_validation(rollouts, ["never_ran"])

        lines = report.text.splitlines()
        assert lines[0] == "task_id\tstatus\treward"
        assert "solved\toracle solved\t1.0" in lines
        assert "wrong\toracle solved (harbor:missing_reward)\t0.0" in lines
        assert "skipped\tunvalidated (no solution/)\t-" in lines
        assert "broken\tfailed: seed exploded\t-" in lines
        assert "masked\tmasked (provider_unavailable)\t-" in lines
        assert "never_ran\tunvalidated (no solution/)\t-" in lines
        assert report.ok is False

    def test_all_good_is_ok(self, tmp_path):
        from nemo_gym.tasks.harbor.validation_report import summarize_validation

        rollouts = tmp_path / "rollouts.jsonl"
        row = {
            "_ng_task_id": {"taskset": "ds", "task_id": "a"},
            "reward": 1.0,
            "response": {"metadata": {"oracle": "solved"}},
        }
        rollouts.write_text(json.dumps(row) + "\n")
        report = summarize_validation(rollouts, [])
        assert report.ok is True and "a\toracle solved\t1.0" in report.text

    @pytest.mark.parametrize("row", [{}, {"response": {}}, {"response": {"metadata": {}}}])
    def test_a_result_without_an_oracle_status_never_passes(self, tmp_path, row):
        """A rollouts file a model run wrote holds reward-1.0 rows with no oracle status; none validates a task."""
        from nemo_gym.tasks.harbor.validation_report import summarize_validation

        rollouts = tmp_path / "rollouts.jsonl"
        rollouts.write_text(
            json.dumps({"_ng_task_id": {"taskset": "ds", "task_id": "a"}, "reward": 1.0, **row}) + "\n"
        )
        report = summarize_validation(rollouts, [])
        assert report.ok is False and "a\toracle missing (not an oracle result)\t1.0" in report.text

    def test_an_unknown_oracle_status_never_passes(self, tmp_path):
        from nemo_gym.tasks.harbor.validation_report import summarize_validation

        rollouts = tmp_path / "rollouts.jsonl"
        row = {
            "_ng_task_id": {"taskset": "ds", "task_id": "a"},
            "reward": 1.0,
            "response": {"metadata": {"oracle": "weird"}},
        }
        rollouts.write_text(json.dumps(row) + "\n")
        report = summarize_validation(rollouts, [])
        assert report.ok is False and "a\toracle status 'weird' (not an oracle result)\t1.0" in report.text

    def test_missing_rollouts_file(self, tmp_path):
        from nemo_gym.tasks.harbor.validation_report import summarize_validation

        report = summarize_validation(tmp_path / "none.jsonl", [])
        assert report.ok is False and "no rollouts written" in report.text
