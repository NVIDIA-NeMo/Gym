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
"""`gym dev compare`: the per-task parity report between an old server's rollouts and the Harbor path's."""

import io
import json
import re
import sys
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

import nemo_gym.cli.dev as cli_dev
import nemo_gym.global_config as gc
from nemo_gym.cli.main import main
from nemo_gym.rollout_compare import (
    CompareInputError,
    RolloutCompareConfig,
    compare_rollouts,
    detect_task_key,
    load_rollouts,
    render_report,
    run_compare,
    summary_dict,
    tail_lines,
)


def old_row(task: str, reward, **extra) -> dict:
    """A legacy-server row: namespaced `task_name`, verifier stdout embedded as `test_output`."""
    row = {
        "task_name": f"terminal-bench/{task}",
        "reward": reward,
        "mask_sample": False,
        "failure_kind": None,
        "error": None,
        "test_output": f"collected 2 items\nPASSED test_a\nFAILED test_b\n1 failed, 1 passed ({task} old)\n",
        "response": {"output": [], "metadata": {}},
    }
    row.update(extra)
    return row


def new_row(task: str, reward, logs_dir: str | None = None, **extra) -> dict:
    """A Harbor-path row: `_ng_task_id` dict, verifier stdout on disk under `verifier_logs_dir`."""
    row = {
        "_ng_task_id": {"taskset": "terminal-bench-2-1", "task_id": task},
        "reward": reward,
        "mask_sample": False,
        "failure_kind": None,
        "verifier_logs_dir": logs_dir,
        "response": {"output": [], "metadata": {"terminus2_outcome": "completed"}},
    }
    row.update(extra)
    return row


def failure_record(task: str, message: str = "502, message='Bad Gateway'") -> dict:
    """A `<name>_failures.jsonl` record as the Harbor path writes it for a masked infrastructure failure."""
    return {
        "_ng_task_id": {"taskset": "terminal-bench-2-1", "task_id": task},
        "_ng_failure_class": "environment_server_failed",
        "_ng_failure_message": message,
    }


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


def with_reward_literal(row: dict, literal: str) -> str:
    """The row as a JSONL line whose `reward` is the raw JSON token `literal` (e.g. `NaN`, `"1.0"`, `true`)."""
    line = json.dumps(row)
    assert line.count('"reward": 0.0') == 1
    return line.replace('"reward": 0.0', f'"reward": {literal}')


def write_verifier_log(root: Path, name: str, lines: list[str]) -> str:
    """Create `<root>/<name>/test-stdout.txt` and return the relative `verifier_logs_dir` a row would carry."""
    (root / name).mkdir(parents=True)
    (root / name / "test-stdout.txt").write_text("\n".join(lines) + "\n")
    return name


class TestClassification:
    def test_identical_tasks(self) -> None:
        result = compare_rollouts([old_row("a", 1.0), old_row("b", 0.0)], [new_row("a", 1.0), new_row("b", 0.0)])
        counts = result.counts()
        assert counts["identical"] == 2
        assert counts["flipped"] == 0 and counts["masked"] == 0 and counts["missing"] == 0
        assert result.old_key == "task_name" and result.new_key == "_ng_task_id"
        # The legacy namespace is stripped so the sides join.
        assert result.old_prefix_stripped == "terminal-bench/" and result.new_prefix_stripped is None

    def test_flip_in_each_direction(self) -> None:
        result = compare_rollouts(
            [old_row("old-wins", 1.0), old_row("new-wins", 0.0)],
            [new_row("old-wins", 0.0), new_row("new-wins", 1.0)],
        )
        counts = result.counts()
        assert counts["flipped"] == 2
        assert counts["flipped_old_win"] == 1 and counts["flipped_new_win"] == 1
        by_task = {t.task: t for t in result.tasks}
        assert by_task["old-wins"].winner == "old" and by_task["old-wins"].loser == "new"
        assert by_task["new-wins"].winner == "new" and by_task["new-wins"].loser == "old"

    @pytest.mark.parametrize(
        "old_extra, new_extra, expected_side, expected_reason",
        [
            ({}, {"mask_sample": True}, "new", "mask_sample"),
            ({"failure_kind": "agent_crashed"}, {}, "old", "failure_kind=agent_crashed"),
            ({}, {"reward": None}, "new", "no reward"),
        ],
    )
    def test_masked_sides(self, old_extra, new_extra, expected_side, expected_reason) -> None:
        # Masking wins over the reward comparison: a masked task is never a flip even when rewards differ.
        old, new = old_row("t", 1.0), new_row("t", 0.0)
        old.update(old_extra)
        new.update(new_extra)
        result = compare_rollouts([old], [new])
        counts = result.counts()
        assert counts["masked"] == 1 and counts["flipped"] == 0 and counts["identical"] == 0
        assert counts[f"masked_{expected_side}"] == 1
        task = result.tasks[0]
        side = task.new if expected_side == "new" else task.old
        assert side.masked == expected_reason

    def test_missing_in_each_direction(self) -> None:
        result = compare_rollouts(
            [old_row("both", 1.0), old_row("old-only", 1.0)], [new_row("both", 1.0), new_row("new-only", 0.0)]
        )
        counts = result.counts()
        assert counts["missing"] == 2
        assert counts["missing_old_only"] == 1 and counts["missing_new_only"] == 1
        by_task = {t.task: t for t in result.tasks}
        assert by_task["old-only"].new is None and by_task["new-only"].old is None

    def test_repeats_compare_on_mean_reward(self) -> None:
        # Two old rows (1.0, 0.0) average to 0.5 against a single new 0.5: identical, and the report says so.
        result = compare_rollouts([old_row("t", 1.0), old_row("t", 0.0)], [new_row("t", 0.5)])
        assert result.counts()["identical"] == 1
        task = result.tasks[0]
        assert task.old.reward == 0.5 and task.old.repeated
        assert task.old.reward_text() == "0.5 (mean of 2 rows)"
        assert result.notes == ["1 task(s) have several rows on a side; they are compared on the mean reward: t"]

    def test_repeats_mean_can_flip(self) -> None:
        result = compare_rollouts([old_row("t", 1.0), old_row("t", 1.0)], [new_row("t", 1.0), new_row("t", 0.0)])
        task = result.tasks[0]
        assert task.category == "flipped" and task.winner == "old"
        assert task.new.reward == 0.5

    def test_agent_error_on_a_flip_is_set_apart(self) -> None:
        # Mirrors audit_v3's "RERUN (infra)" verdict: rewards differ but the losing side timed out in the agent.
        new = new_row("t", 0.0)
        new["response"]["metadata"]["terminus2_error_type"] = "TimeoutError"
        result = compare_rollouts([old_row("t", 1.0)], [new])
        counts = result.counts()
        assert counts["flipped"] == 0 and counts["flipped_agent_error"] == 1
        assert result.tasks[0].new.agent_error == "terminus2_error_type=TimeoutError"

    def test_legacy_error_field_is_an_agent_error(self) -> None:
        old = old_row("t", 0.0, error="Traceback (most recent call last):\n  ...\nClientResponseError: 500")
        result = compare_rollouts([old], [new_row("t", 1.0)])
        assert result.tasks[0].category == "flipped_agent_error"
        assert result.tasks[0].old.agent_error == "error: ClientResponseError: 500"

    def test_agent_error_with_equal_rewards_stays_identical(self) -> None:
        new = new_row("t", 1.0)
        new["response"]["metadata"]["terminus2_error_type"] = "TimeoutError"
        assert compare_rollouts([old_row("t", 1.0)], [new]).tasks[0].category == "identical"

    def test_agent_error_on_the_winning_side_is_also_set_apart(self) -> None:
        # The error is on the side that scored higher: still a rerun candidate, not a counted new-win.
        new = new_row("t", 1.0)
        new["response"]["metadata"]["terminus2_error_type"] = "TimeoutError"
        result = compare_rollouts([old_row("t", 0.0)], [new])
        task = result.tasks[0]
        assert task.category == "flipped_agent_error" and task.winner == "new" and task.loser == "old"
        counts = result.counts()
        assert counts["flipped"] == 0 and counts["flipped_new_win"] == 0 and counts["flipped_agent_error"] == 1

    @pytest.mark.parametrize("literal, shown", [("NaN", "nan"), ("Infinity", "inf"), ("-Infinity", "-inf")])
    def test_non_finite_reward_is_masked_not_a_new_win(self, tmp_path: Path, literal: str, shown: str) -> None:
        # json.loads accepts the NaN/Infinity tokens, so a broken producer can emit them; `nan > 0.0` is False
        # and `inf > 0.0` is True, so without the rule they would land in the flip counts.
        path = tmp_path / "new.jsonl"
        path.write_text(with_reward_literal(new_row("t", 0.0), literal) + "\n")
        result = compare_rollouts([old_row("t", 0.0)], load_rollouts(path))
        task = result.tasks[0]
        assert task.category == "masked"
        assert task.new.masked == f"non-finite reward ({shown})" and task.new.reward is None
        counts = result.counts()
        assert counts["flipped"] == 0 and counts["flipped_new_win"] == 0 and counts["masked_new"] == 1
        assert f"- t: old=0.0 new=none; new non-finite reward ({shown})" in render_report(result, roots=[])
        assert json.loads(json.dumps(summary_dict(result)))["tasks"][0]["new"]["reward"] is None

    def test_non_numeric_reward_is_an_input_error_not_an_assert(self) -> None:
        with pytest.raises(CompareInputError, match="reward '1.0' is not a number"):
            compare_rollouts([old_row("t", "1.0")], [new_row("t", 1.0)])
        with pytest.raises(CompareInputError, match="reward True is not a number"):
            compare_rollouts([old_row("t", 1.0)], [new_row("t", True)])

    def test_repeated_rows_with_one_masked_are_masked_not_flipped(self) -> None:
        # Two clean 1.0 rows and one masked row against old 0.0: unmasked this would be a new-win flip.
        new_rows = [
            new_row("t", 1.0),
            new_row("t", 1.0),
            new_row("t", None, mask_sample=True, failure_kind="environment_server_failed"),
        ]
        result = compare_rollouts([old_row("t", 0.0)], new_rows)
        task = result.tasks[0]
        assert task.category == "masked"
        assert task.new.masked == "1/3 rows masked: mask_sample (failure_kind=environment_server_failed)"
        assert task.new.reward == 1.0 and task.new.reward_text() == "1.0 (mean of 2 of 3 rows)"
        counts = result.counts()
        assert counts["masked"] == 1 and counts["flipped"] == 0 and counts["flipped_agent_error"] == 0
        report = render_report(result, roots=[])
        assert "## Flips (0)" in report
        assert (
            "## Masked (1)\n- t: old=0.0 new=1.0 (mean of 2 of 3 rows); "
            "new 1/3 rows masked: mask_sample (failure_kind=environment_server_failed)"
        ) in report

    def test_both_sides_empty_is_no_rows(self) -> None:
        with pytest.raises(CompareInputError, match="^no rows in either input$"):
            compare_rollouts([], [])


class TestTaskKey:
    def test_detects_verifier_metadata_task_id(self) -> None:
        rows = [{"verifier_metadata": {"task_id": "x"}, "reward": 1.0}]
        assert detect_task_key(rows) == "verifier_metadata.task_id"

    def test_explicit_key_applies_to_both_sides(self) -> None:
        old = [{"verifier_metadata": {"instance_id": "repo-1"}, "task_name": "other", "reward": 1.0}]
        new = [{"verifier_metadata": {"instance_id": "repo-1"}, "_ng_task_id": "ignored", "reward": 1.0}]
        result = compare_rollouts(old, new, key="verifier_metadata.instance_id")
        assert result.old_key == result.new_key == "verifier_metadata.instance_id"
        assert result.counts()["identical"] == 1

    def test_no_key_is_an_input_error(self) -> None:
        with pytest.raises(CompareInputError, match="no task id field found"):
            compare_rollouts([{"reward": 1.0}], [new_row("t", 1.0)])

    def test_explicit_key_missing_on_a_row_is_an_input_error(self) -> None:
        with pytest.raises(CompareInputError, match="row has no 'nope' field"):
            compare_rollouts([old_row("t", 1.0)], [new_row("t", 1.0)], key="nope")

    def test_string_ng_task_id_is_accepted(self) -> None:
        result = compare_rollouts([old_row("t", 1.0)], [new_row("t", 1.0, _ng_task_id="t")])
        assert result.counts()["identical"] == 1

    def test_prefix_is_stripped_from_whichever_side_carries_it(self) -> None:
        plain_old = [{"task_name": "t", "reward": 1.0}]
        result = compare_rollouts(plain_old, [new_row("t", 1.0, _ng_task_id="tb/t")])
        assert result.old_prefix_stripped is None and result.new_prefix_stripped == "tb/"
        assert result.counts()["identical"] == 1

    def test_same_task_id_in_two_tasksets_is_keyed_by_taskset(self) -> None:
        def row(taskset: str, reward: float) -> dict:
            return new_row("x", reward, _ng_task_id={"taskset": taskset, "task_id": "x"})

        result = compare_rollouts([row("tb-a", 1.0), row("tb-b", 0.0)], [row("tb-a", 1.0), row("tb-b", 1.0)])
        assert result.keyed_by_taskset
        assert [t.task for t in result.tasks] == ["tb-a/x", "tb-b/x"]
        counts = result.counts()
        assert counts["identical"] == 1 and counts["flipped_new_win"] == 1 and counts["missing"] == 0
        assert result.notes == [
            "a side spans several tasksets (tb-a, tb-b); tasks are keyed and listed as taskset/task_id"
        ]
        assert "### tb-b/x: old=0.0 new=1.0 -> new wins" in render_report(result, roots=[])
        assert summary_dict(result)["join"]["keyed_by_taskset"] is True

    def test_several_tasksets_on_one_side_only_still_keys_by_taskset(self) -> None:
        def row(taskset: str, reward: float) -> dict:
            return new_row("x", reward, _ng_task_id={"taskset": taskset, "task_id": "x"})

        result = compare_rollouts([row("tb-a", 1.0), row("tb-b", 0.0)], [row("tb-a", 1.0)])
        assert [(t.task, t.category) for t in result.tasks] == [("tb-a/x", "identical"), ("tb-b/x", "missing")]

    def test_single_taskset_keeps_the_plain_task_id(self) -> None:
        result = compare_rollouts([old_row("x", 1.0)], [new_row("x", 1.0)])
        assert not result.keyed_by_taskset and result.tasks[0].task == "x"
        assert summary_dict(result)["join"]["keyed_by_taskset"] is False

    def test_both_sides_prefixed_differently_are_not_joined(self) -> None:
        # Stripping both would silently match unrelated namespaces; the tasks stay missing instead.
        result = compare_rollouts([old_row("t", 1.0)], [new_row("t", 1.0, _ng_task_id="other-bench/t")])
        assert result.old_prefix_stripped is None and result.new_prefix_stripped is None
        assert result.counts()["missing"] == 2


class TestLoading:
    def test_failures_sidecar_rows_are_folded_in_as_masked(self, tmp_path: Path) -> None:
        # The Harbor path writes masked infrastructure failures next to the rollouts, not as rows.
        path = write_jsonl(tmp_path / "new.jsonl", [new_row("ok", 1.0)])
        write_jsonl(
            tmp_path / "new_failures.jsonl",
            [
                {
                    "_ng_task_id": {"taskset": "terminal-bench-2-1", "task_id": "broken"},
                    "_ng_failure_class": "environment_server_failed",
                    "_ng_failure_message": "502, message='Bad Gateway'",
                }
            ],
        )
        rows = load_rollouts(path)
        assert len(rows) == 2
        failure = rows[1]
        assert failure["mask_sample"] is True and failure["reward"] is None
        assert failure["failure_kind"] == "environment_server_failed"
        result = compare_rollouts([old_row("ok", 1.0), old_row("broken", 0.0)], rows)
        assert result.counts() == {
            "tasks": 2,
            "identical": 1,
            "flipped": 0,
            "flipped_old_win": 0,
            "flipped_new_win": 0,
            "flipped_agent_error": 0,
            "masked": 1,
            "masked_old": 0,
            "masked_new": 1,
            "missing": 0,
            "missing_old_only": 0,
            "missing_new_only": 0,
        }
        assert result.new_masked_rows == 1

    def test_blank_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        path.write_text(json.dumps(new_row("t", 1.0)) + "\n\n\n")
        assert len(load_rollouts(path)) == 1

    def test_invalid_json_names_the_line(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        path.write_text('{"ok": 1}\nnot json\n')
        with pytest.raises(CompareInputError, match=r"rows\.jsonl:2: invalid JSON"):
            load_rollouts(path)

    def test_non_object_line_is_rejected(self, tmp_path: Path) -> None:
        path = tmp_path / "rows.jsonl"
        path.write_text("[1, 2]\n")
        with pytest.raises(CompareInputError, match="expected a JSON object"):
            load_rollouts(path)

    def test_missing_file_is_an_input_error(self, tmp_path: Path) -> None:
        with pytest.raises(CompareInputError, match="cannot read"):
            load_rollouts(tmp_path / "nope.jsonl")

    def test_line_separator_inside_a_string_field_does_not_split_the_row(self, tmp_path: Path) -> None:
        # str.splitlines also breaks at U+2028, U+2029 and U+0085, which a verifier log may well contain; the file
        # has one JSON object per "\n"-terminated line and must load as one row.
        text = "before\u2028after\u2029end\u0085tail"
        path = tmp_path / "rows.jsonl"
        path.write_text(json.dumps(new_row("t", 1.0, test_output=text), ensure_ascii=False) + "\n", encoding="utf-8")
        rows = load_rollouts(path)
        assert len(rows) == 1 and rows[0]["test_output"] == text

    @pytest.mark.parametrize("literal, shown", [('"1.0"', "'1.0'"), ("true", "True"), ("false", "False")])
    def test_non_numeric_reward_names_file_line_and_task(self, tmp_path: Path, literal: str, shown: str) -> None:
        # A numeric string is not coerced: the producer is broken and the report must not guess.
        path = tmp_path / "new.jsonl"
        path.write_text(
            json.dumps(new_row("fine", 1.0)) + "\n" + with_reward_literal(new_row("broken", 0.0), literal) + "\n"
        )
        with pytest.raises(
            CompareInputError,
            match=rf"new\.jsonl:2: reward {re.escape(shown)} is not a number \(task terminal-bench-2-1/broken\)$",
        ):
            load_rollouts(path)


class TestReport:
    def test_summary_table_and_flip_tail_from_log_file(self, tmp_path: Path) -> None:
        logs_dir = write_verifier_log(tmp_path, "verifier/session-1", [f"line {i}" for i in range(1, 21)])
        result = compare_rollouts(
            [old_row("a", 1.0), old_row("b", 1.0)], [new_row("a", 1.0), new_row("b", 0.0, logs_dir=logs_dir)]
        )
        report = render_report(result, roots=[tmp_path], tail=3)
        lines = report.splitlines()
        assert lines[0] == "gym dev compare: 2 tasks (old 2 rows, new 2 rows)"
        assert lines[1] == "join key: old=task_name (prefix 'terminal-bench/' stripped), new=_ng_task_id"
        assert "identical                    1" in lines
        assert "flipped                      1   old-win 1, new-win 0" in lines
        assert "masked                       0   old 0, new 0" in lines
        assert "missing                      0   old-only 0, new-only 0" in lines
        assert "## Flips (1)" in lines
        assert "### b: old=1.0 new=0.0 -> old wins" in lines
        log_path = tmp_path / logs_dir / "test-stdout.txt"
        assert f"    new verifier output (last 3 lines of {log_path}):" in lines
        # Exactly the last three lines of the losing side's log, nothing from the winner.
        assert [line for line in lines if line.startswith("    | ")] == [
            "    | line 18",
            "    | line 19",
            "    | line 20",
        ]
        assert "## Masked (0)" in lines and "## Missing (0)" in lines

    def test_flip_tail_from_embedded_old_output_and_default_tail(self, tmp_path: Path) -> None:
        # New wins, so the old side's embedded `test_output` is shown; default tail is 15 lines.
        result = compare_rollouts([old_row("t", 0.0)], [new_row("t", 1.0)])
        config_default = RolloutCompareConfig(old_rollouts="o", new_rollouts="n")
        assert config_default.tail == 15
        report = render_report(result, roots=[tmp_path], tail=config_default.tail)
        assert "### t: old=0.0 new=1.0 -> new wins" in report
        assert "    old verifier output (last 15 lines of row field 'test_output'):" in report
        assert "    | 1 failed, 1 passed (t old)" in report
        assert "    | collected 2 items" in report  # the whole 4-line output fits under 15

    def test_flip_without_any_verifier_output_says_unavailable(self, tmp_path: Path) -> None:
        result = compare_rollouts([old_row("t", 1.0)], [new_row("t", 0.0, logs_dir="results/verifier/gone")])
        report = render_report(result, roots=[tmp_path / "root-a", tmp_path / "root-b"], tail=15)
        assert (
            f"    new verifier output unavailable: results/verifier/gone not found under "
            f"{tmp_path / 'root-a'}, {tmp_path / 'root-b'}"
        ) in report
        result = compare_rollouts([old_row("t", 1.0)], [new_row("t", 0.0)])
        report = render_report(result, roots=[tmp_path], tail=15)
        assert "    new verifier output unavailable: row has no verifier_logs_dir and no embedded verifier output" in (
            report
        )

    def test_absolute_logs_dir_and_file_path_are_read_directly(self, tmp_path: Path) -> None:
        logs_dir = tmp_path / write_verifier_log(tmp_path, "abs", ["only line"])
        result = compare_rollouts([old_row("t", 1.0)], [new_row("t", 0.0, logs_dir=str(logs_dir))])
        assert "    | only line" in render_report(result, roots=[], tail=5)
        file_path = logs_dir / "test-stdout.txt"
        result = compare_rollouts([old_row("t", 1.0)], [new_row("t", 0.0, logs_dir=str(file_path))])
        assert f"(last 5 lines of {file_path})" in render_report(result, roots=[], tail=5)

    def test_agent_error_flips_masked_and_missing_sections(self, tmp_path: Path) -> None:
        timed_out = new_row("timeout", 0.0)
        timed_out["response"]["metadata"]["terminus2_error_type"] = "TimeoutError"
        result = compare_rollouts(
            [old_row("timeout", 1.0), old_row("masked", 0.0), old_row("old-only", 1.0)],
            [timed_out, new_row("masked", None, mask_sample=True, failure_kind="environment_server_failed")],
        )
        report = render_report(result, roots=[tmp_path], tail=2)
        assert "flipped (agent error)        1   rewards differ and a side reports an agent error" in report
        assert "## Flips with agent errors (1)" in report
        assert "### timeout: old=1.0 new=0.0 -> old wins\n    new agent error: terminus2_error_type=TimeoutError" in (
            report
        )
        assert (
            "## Masked (1)\n- masked: old=0.0 new=none; new mask_sample (failure_kind=environment_server_failed)"
            in (report)
        )
        assert "## Missing (1)\n- old-only: only on old side (reward 1.0)" in report

    def test_unreadable_verifier_log_is_reported_and_the_report_continues(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        logs_dir = write_verifier_log(tmp_path, "locked", ["must not leak"])
        log_path = tmp_path / logs_dir / "test-stdout.txt"
        real_read_text = Path.read_text

        def locked_read_text(self: Path, *args, **kwargs) -> str:
            if self == log_path:
                raise PermissionError(13, "Permission denied")
            return real_read_text(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", locked_read_text)
        result = compare_rollouts(
            [old_row("a", 1.0), old_row("b", 1.0)], [new_row("a", 0.0, logs_dir=logs_dir), new_row("b", 0.0)]
        )
        report = render_report(result, roots=[tmp_path], tail=5)
        assert f"    new verifier output unavailable: cannot read {log_path}: Permission denied" in report
        assert "### b: old=1.0 new=0.0 -> old wins" in report  # the report goes on to the next flip
        assert "must not leak" not in report

    def test_repeats_choose_the_worst_losing_row_for_the_tail(self, tmp_path: Path) -> None:
        good = write_verifier_log(tmp_path, "good", ["all passed"])
        bad = write_verifier_log(tmp_path, "bad", ["1 failed"])
        result = compare_rollouts(
            [old_row("t", 1.0)], [new_row("t", 1.0, logs_dir=good), new_row("t", 0.0, logs_dir=bad)]
        )
        report = render_report(result, roots=[tmp_path], tail=5)
        assert "### t: old=1.0 new=0.5 (mean of 2 rows) -> old wins" in report
        assert "    | 1 failed" in report and "all passed" not in report

    def test_tail_lines(self) -> None:
        assert tail_lines("a\nb\nc\n", 2) == ["b", "c"]
        assert tail_lines("a\nb", 10) == ["a", "b"]
        assert tail_lines("a\nb", 0) == []

    def test_identical_task_has_no_winner_or_loser(self) -> None:
        task = compare_rollouts([old_row("t", 1.0)], [new_row("t", 1.0)]).tasks[0]
        assert task.winner is None and task.loser is None

    def test_summary_dict_missing_side_is_null(self) -> None:
        # An empty new side borrows the old side's key; with nothing to join against, no prefix is stripped.
        summary = summary_dict(compare_rollouts([old_row("t", 1.0)], []))
        assert summary["tasks"] == [
            {
                "task": "terminal-bench/t",
                "category": "missing",
                "winner": None,
                "old": {"reward": 1.0, "rows": 1, "masked": None, "agent_error": None},
                "new": None,
            }
        ]

    def test_summary_dict_is_json_serialisable_and_complete(self) -> None:
        result = compare_rollouts([old_row("a", 1.0), old_row("b", 1.0)], [new_row("a", 1.0), new_row("b", 0.0)])
        summary = json.loads(json.dumps(summary_dict(result)))
        assert summary["counts"]["identical"] == 1 and summary["counts"]["flipped_old_win"] == 1
        assert summary["join"] == {
            "old_key": "task_name",
            "new_key": "_ng_task_id",
            "old_prefix_stripped": "terminal-bench/",
            "new_prefix_stripped": None,
            "keyed_by_taskset": False,
        }
        flip = next(t for t in summary["tasks"] if t["task"] == "b")
        assert flip == {
            "task": "b",
            "category": "flipped",
            "winner": "old",
            "old": {"reward": 1.0, "rows": 1, "masked": None, "agent_error": None},
            "new": {"reward": 0.0, "rows": 1, "masked": None, "agent_error": None},
        }


class TestRunCompare:
    def test_writes_report_and_json(self, tmp_path: Path) -> None:
        old = write_jsonl(tmp_path / "old.jsonl", [old_row("a", 1.0), old_row("b", 0.0)])
        new = write_jsonl(tmp_path / "new.jsonl", [new_row("a", 1.0), new_row("b", 1.0)])
        out = io.StringIO()
        json_path = tmp_path / "summary.json"
        config = RolloutCompareConfig(old_rollouts=str(old), new_rollouts=str(new), json_output=str(json_path), tail=1)
        assert run_compare(config, out=out) == 0
        text = out.getvalue()
        assert "flipped                      1   old-win 0, new-win 1" in text
        assert "    | 1 failed, 1 passed (b old)" in text
        assert text.endswith(f"wrote {json_path}\n")
        assert json.loads(json_path.read_text())["counts"]["flipped_new_win"] == 1

    def test_logs_root_resolves_relative_verifier_dirs(self, tmp_path: Path) -> None:
        server_dir = tmp_path / "server"
        logs_dir = write_verifier_log(server_dir, "results/verifier/s1", ["FAILED test_x", "1 failed"])
        old = write_jsonl(tmp_path / "old.jsonl", [old_row("a", 1.0)])
        new = write_jsonl(tmp_path / "new.jsonl", [new_row("a", 0.0, logs_dir=logs_dir)])
        out = io.StringIO()
        config = RolloutCompareConfig(old_rollouts=str(old), new_rollouts=str(new), logs_root=str(server_dir))
        assert run_compare(config, out=out) == 0
        assert "    | FAILED test_x\n    | 1 failed\n" in out.getvalue()

    def test_unreadable_input_exits_nonzero(self, tmp_path: Path, capsys: CaptureFixture) -> None:
        new = write_jsonl(tmp_path / "new.jsonl", [new_row("a", 1.0)])
        config = RolloutCompareConfig(old_rollouts=str(tmp_path / "missing.jsonl"), new_rollouts=str(new))
        assert run_compare(config, out=io.StringIO()) == 1
        assert "error: cannot read" in capsys.readouterr().err

    def test_no_shared_key_exits_nonzero(self, tmp_path: Path, capsys: CaptureFixture) -> None:
        old = write_jsonl(tmp_path / "old.jsonl", [{"reward": 1.0}])
        new = write_jsonl(tmp_path / "new.jsonl", [new_row("a", 1.0)])
        config = RolloutCompareConfig(old_rollouts=str(old), new_rollouts=str(new))
        assert run_compare(config, out=io.StringIO()) == 1
        assert "no task id field found" in capsys.readouterr().err

    def test_reference_fixture_shape(self, tmp_path: Path) -> None:
        # The parity-v3 gate in miniature: every category the real report produced, asserted as one counts dict.
        err_on_loser = new_row("err-loser", 0.0)  # new loses and reports the agent error
        err_on_loser["response"]["metadata"]["terminus2_error_type"] = "TimeoutError"
        err_on_winner = new_row("err-winner", 1.0)  # new wins but reports the agent error
        err_on_winner["response"]["metadata"]["terminus2_error_type"] = "TimeoutError"
        old = write_jsonl(
            tmp_path / "old.jsonl",
            [
                old_row("same-1", 1.0),
                old_row("same-0", 0.0),
                old_row("old-wins", 1.0),
                old_row("new-wins", 0.0),
                old_row("err-loser", 1.0),
                old_row("err-winner", 0.0),
                old_row("infra", 0.0),
                # An old-side agent error with reward 0.0 against a sidecar-masked new side: masked, not a flip.
                old_row("infra-old-error", 0.0, error="Traceback (most recent call last):\n  ...\nTimeoutError"),
            ],
        )
        new = write_jsonl(
            tmp_path / "new.jsonl",
            [
                new_row("same-1", 1.0),
                new_row("same-0", 0.0),
                new_row("old-wins", 0.0),
                new_row("new-wins", 1.0),
                err_on_loser,
                err_on_winner,
            ],
        )
        write_jsonl(tmp_path / "new_failures.jsonl", [failure_record("infra"), failure_record("infra-old-error")])
        json_path = tmp_path / "summary.json"
        out = io.StringIO()
        config = RolloutCompareConfig(old_rollouts=str(old), new_rollouts=str(new), json_output=str(json_path))
        assert run_compare(config, out=out) == 0
        summary = json.loads(json_path.read_text())
        assert summary["counts"] == {
            "tasks": 8,
            "identical": 2,
            "flipped": 2,
            "flipped_old_win": 1,
            "flipped_new_win": 1,
            "flipped_agent_error": 2,
            "masked": 2,
            "masked_old": 0,
            "masked_new": 2,
            "missing": 0,
            "missing_old_only": 0,
            "missing_new_only": 0,
        }
        by_task = {t["task"]: t for t in summary["tasks"]}
        assert by_task["err-loser"]["winner"] == "old" and by_task["err-winner"]["winner"] == "new"
        assert by_task["infra-old-error"]["category"] == "masked"
        assert by_task["infra-old-error"]["old"]["agent_error"] == "error: TimeoutError"
        text = out.getvalue()
        assert "## Flips (2)" in text and "## Flips with agent errors (2)" in text and "## Masked (2)" in text
        assert "infra-old-error" not in text.split("## Masked")[0]  # listed under masked only

    def test_relative_logs_dir_resolves_against_each_files_own_directory(
        self, tmp_path: Path, monkeypatch: MonkeyPatch
    ) -> None:
        old_dir, new_dir = tmp_path / "old-run", tmp_path / "new-run"
        old_log = write_verifier_log(old_dir, "logs/o", ["old side failed"])
        new_log = write_verifier_log(new_dir, "logs/n", ["new side failed"])
        # The embedded output is blanked so only the file under the row's own directory can supply the tail.
        old = write_jsonl(
            old_dir / "old.jsonl",
            [old_row("a", 1.0), old_row("b", 0.0, test_output="", verifier_logs_dir=old_log)],
        )
        new = write_jsonl(new_dir / "new.jsonl", [new_row("a", 0.0, logs_dir=new_log), new_row("b", 1.0)])
        (tmp_path / "elsewhere").mkdir()
        monkeypatch.chdir(tmp_path / "elsewhere")
        out = io.StringIO()
        assert run_compare(RolloutCompareConfig(old_rollouts=str(old), new_rollouts=str(new)), out=out) == 0
        text = out.getvalue()
        assert "    | new side failed" in text and "    | old side failed" in text
        assert "unavailable" not in text

    def test_two_empty_inputs_say_no_rows(self, tmp_path: Path, capsys: CaptureFixture) -> None:
        old, new = tmp_path / "old.jsonl", tmp_path / "new.jsonl"
        old.write_text("")
        new.write_text("\n\n")
        assert run_compare(RolloutCompareConfig(old_rollouts=str(old), new_rollouts=str(new)), out=io.StringIO()) == 1
        err = capsys.readouterr().err
        assert err == "error: no rows in either input\n"

    def test_non_numeric_reward_exits_nonzero_naming_the_row(self, tmp_path: Path, capsys: CaptureFixture) -> None:
        old = write_jsonl(tmp_path / "old.jsonl", [old_row("a", 1.0)])
        new = tmp_path / "new.jsonl"
        new.write_text(with_reward_literal(new_row("a", 0.0), '"1.0"') + "\n")
        assert run_compare(RolloutCompareConfig(old_rollouts=str(old), new_rollouts=str(new)), out=io.StringIO()) == 1
        assert capsys.readouterr().err == (
            f"error: {new}:1: reward '1.0' is not a number (task terminal-bench-2-1/a)\n"
        )

    def test_unwritable_json_path_is_an_error(self, tmp_path: Path, capsys: CaptureFixture) -> None:
        old = write_jsonl(tmp_path / "old.jsonl", [old_row("a", 1.0)])
        new = write_jsonl(tmp_path / "new.jsonl", [new_row("a", 1.0)])
        json_path = tmp_path / "missing-dir" / "summary.json"
        config = RolloutCompareConfig(old_rollouts=str(old), new_rollouts=str(new), json_output=str(json_path))
        out = io.StringIO()
        assert run_compare(config, out=out) == 1
        assert capsys.readouterr().err == f"error: cannot write {json_path}: No such file or directory\n"
        assert out.getvalue() == ""  # nothing half-printed
        assert not json_path.parent.exists()  # the directory is not created silently

    def test_json_stdout_prints_the_summary_instead_of_the_report(
        self, tmp_path: Path, capsys: CaptureFixture
    ) -> None:
        old = write_jsonl(tmp_path / "old.jsonl", [old_row("a", 1.0), old_row("b", 0.0)])
        new = write_jsonl(tmp_path / "new.jsonl", [new_row("a", 1.0), new_row("b", 1.0)])
        json_path = tmp_path / "summary.json"
        config = RolloutCompareConfig(
            old_rollouts=str(old), new_rollouts=str(new), json_stdout=True, json_output=str(json_path)
        )
        out = io.StringIO()
        assert run_compare(config, out=out) == 0
        summary = json.loads(out.getvalue())  # stdout is exactly the JSON object
        assert summary["counts"]["flipped_new_win"] == 1
        assert json.loads(json_path.read_text()) == summary
        assert f"wrote {json_path}" in capsys.readouterr().err


class TestCliWiring:
    @staticmethod
    def _capture(monkeypatch: MonkeyPatch) -> dict:
        """Replace `dev_compare` with a recorder of the keyword values it receives and the argv left for Hydra."""
        captured: dict = {}

        def fake_dev_compare(**values) -> None:
            captured.update(values)
            captured["hydra_argv"] = sys.argv[1:]

        monkeypatch.setattr(cli_dev, "dev_compare", fake_dev_compare)
        return captured

    @pytest.mark.parametrize(
        "old_path",
        [
            "old dir/old.jsonl",  # a space
            "résumé/ancien.jsonl",  # non-ASCII, which json.dumps would turn into \u00e9 escapes
            r"runs\old\old.jsonl",  # backslashes, which Hydra's quoted grammar would keep doubled
            'say "hi".jsonl',  # a quote
        ],
    )
    def test_paths_and_quoted_options_reach_the_command_verbatim(
        self, monkeypatch: MonkeyPatch, old_path: str
    ) -> None:
        # Like `gym eval run`'s TARGET, the paths bypass Hydra's override grammar; only --tail travels as an override.
        captured = self._capture(monkeypatch)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gym",
                "dev",
                "compare",
                old_path,
                "new.jsonl",
                "--key",
                "verifier_metadata.task_id",
                "--tail",
                "3",
                "--logs-root",
                "resources_servers/harbor",
                "--json",
                "out.json",
            ],
        )
        main()
        assert captured == {
            "json_stdout": False,
            "old_rollouts": old_path,
            "new_rollouts": "new.jsonl",
            "key": "verifier_metadata.task_id",
            "logs_root": "resources_servers/harbor",
            "json_output": "out.json",
            "hydra_argv": ["+tail=3"],
        }

    def test_optional_flags_default_to_none(self, monkeypatch: MonkeyPatch) -> None:
        captured = self._capture(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["gym", "dev", "compare", "a.jsonl", "b.jsonl"])
        main()
        assert captured == {
            "json_stdout": False,
            "old_rollouts": "a.jsonl",
            "new_rollouts": "b.jsonl",
            "key": None,
            "logs_root": None,
            "json_output": None,
            "hydra_argv": [],
        }

    def test_root_json_toggle_is_passed_on(self, monkeypatch: MonkeyPatch) -> None:
        # `gym --json dev compare` used to be accepted and silently dropped.
        captured = self._capture(monkeypatch)
        monkeypatch.setattr(sys, "argv", ["gym", "--json", "dev", "compare", "a.jsonl", "b.jsonl", "--json", "o.json"])
        main()
        assert captured["json_stdout"] is True and captured["json_output"] == "o.json"

    def test_missing_positional_is_a_usage_error(self, monkeypatch: MonkeyPatch, capsys: CaptureFixture) -> None:
        monkeypatch.setattr(sys, "argv", ["gym", "dev", "compare", "only-one.jsonl"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        assert "NEW_ROLLOUTS" in capsys.readouterr().err

    def test_end_to_end_through_hydra(self, monkeypatch: MonkeyPatch, tmp_path: Path, capsys: CaptureFixture) -> None:
        # The real path: argparse values seed the config, Hydra parses the remaining overrides, `dev_compare`
        # validates `RolloutCompareConfig`. The run lives in a directory with a space and non-ASCII characters.
        run_dir = tmp_path / "résumé dir"
        old = write_jsonl(run_dir / "old.jsonl", [old_row("a", 1.0), old_row("b", 1.0), old_row("c", 0.0)])
        new = write_jsonl(run_dir / "new.jsonl", [new_row("a", 1.0), new_row("b", 0.0), new_row("c", 0.0)])
        json_path = run_dir / "summary.json"
        monkeypatch.setattr(gc, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(
            sys, "argv", ["gym", "dev", "compare", str(old), str(new), "--tail", "1", "--json", str(json_path)]
        )
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "gym dev compare: 3 tasks (old 3 rows, new 3 rows)" in out
        assert "identical                    2" in out
        assert "flipped                      1   old-win 1, new-win 0" in out
        assert "### b: old=1.0 new=0.0 -> old wins" in out
        assert out.endswith(f"wrote {json_path}\n")
        assert json.loads(json_path.read_text())["counts"]["identical"] == 2

    @pytest.mark.parametrize("argv_prefix, argv_suffix", [(["--json"], []), ([], ["+json=true"])])
    def test_root_json_toggle_prints_the_summary_on_stdout(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, capsys: CaptureFixture, argv_prefix: list, argv_suffix: list
    ) -> None:
        old = write_jsonl(tmp_path / "old.jsonl", [old_row("a", 1.0), old_row("b", 1.0)])
        new = write_jsonl(tmp_path / "new.jsonl", [new_row("a", 1.0), new_row("b", 0.0)])
        monkeypatch.setattr(gc, "_GLOBAL_CONFIG_DICT", None)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["gym", *argv_prefix, "dev", "compare", str(old), str(new), *argv_suffix])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 0
        summary = json.loads(capsys.readouterr().out)  # the whole of stdout is the JSON summary; no report
        assert summary["counts"]["flipped_old_win"] == 1
        assert summary["join"]["old_key"] == "task_name"
