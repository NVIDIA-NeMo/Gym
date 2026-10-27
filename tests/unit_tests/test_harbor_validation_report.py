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
        assert lines[0] == "task_id\tstatus\treward\tverify_s\tnetwork"
        assert "solved\toracle solved\t1.0\t-\t-" in lines
        assert "wrong\toracle solved (harbor:missing_reward)\t0.0\t-\t-" in lines
        assert "skipped\tunvalidated (no solution/)\t-\t-\t-" in lines
        assert "broken\tfailed: seed exploded\t-\t-\t-" in lines
        assert "masked\tmasked (provider_unavailable)\t-\t-\t-" in lines
        assert "never_ran\tunvalidated (no solution/)\t-\t-\t-" in lines
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

    def test_an_empty_selection_never_passes(self, tmp_path):
        """No selected task had a solution: nothing ran, and `gym dataset validate X && publish` must stop."""
        from nemo_gym.tasks.harbor.validation_report import summarize_validation

        report = summarize_validation(tmp_path / "rollouts.jsonl", ["a", "b"], expected=[])
        assert report.ok is False
        assert "-\t0 tasks validated (no selected task has a solution/)\t-\t-\t-" in report.text
        assert "a\tunvalidated (no solution/)" in report.text and "b\tunvalidated (no solution/)" in report.text


class TestValidationRows:
    """`_validation_rows` reads a file the collector may still have to repair."""

    @staticmethod
    def record(task_id):
        return {"_ng_task_id": {"taskset": "ds", "task_id": task_id}, "reward": 1.0}

    def test_an_interrupted_final_append_is_skipped(self, tmp_path):
        from nemo_gym.tasks.harbor.validation_report import _validation_rows

        path = tmp_path / "rollouts.jsonl"
        path.write_text(json.dumps(self.record("a")) + "\n" + json.dumps(self.record("b"))[:-7])
        assert _validation_rows(path) == [self.record("a")]

    def test_a_cut_inside_a_multibyte_character_is_skipped(self, tmp_path):
        """A kill can land one byte into a UTF-8 sequence: the tail neither decodes nor parses, and is skipped."""
        from nemo_gym.tasks.harbor.validation_report import _validation_rows

        path = tmp_path / "rollouts.jsonl"
        second = json.dumps({**self.record("b"), "output": "日本語"}, ensure_ascii=False).encode()
        cut = second.index("日".encode()) + 1
        path.write_bytes(json.dumps(self.record("a")).encode() + b"\n" + second[:cut])
        assert _validation_rows(path) == [self.record("a")]

    def test_bytes_that_are_not_utf8_inside_the_file_are_reported_by_line(self, tmp_path):
        from nemo_gym.config_types import ConfigError
        from nemo_gym.tasks.harbor.validation_report import _validation_rows

        path = tmp_path / "rollouts.jsonl"
        path.write_bytes(json.dumps(self.record("a")).encode() + b"\n" + "日".encode()[:1] + b"\n")
        with pytest.raises(ConfigError, match=r"line 2: not a JSON record \('utf-8' codec can't decode"):
            _validation_rows(path)

    def test_a_complete_final_record_without_a_newline_is_read(self, tmp_path):
        from nemo_gym.tasks.harbor.validation_report import _validation_rows

        path = tmp_path / "rollouts.jsonl"
        path.write_text(json.dumps(self.record("a")) + "\n" + json.dumps(self.record("b")))
        assert _validation_rows(path) == [self.record("a"), self.record("b")]

    def test_a_damaged_interior_record_is_reported_by_line(self, tmp_path):
        from nemo_gym.config_types import ConfigError
        from nemo_gym.tasks.harbor.validation_report import _validation_rows

        path = tmp_path / "rollouts.jsonl"
        path.write_text(json.dumps(self.record("a")) + '\n{"half\n' + json.dumps(self.record("b")) + "\n")
        with pytest.raises(ConfigError, match=rf"{path}, line 2: not a JSON record"):
            _validation_rows(path)

    def test_a_missing_file_has_no_rows(self, tmp_path):
        from nemo_gym.tasks.harbor.validation_report import _validation_rows

        assert _validation_rows(tmp_path / "none.jsonl") == []
