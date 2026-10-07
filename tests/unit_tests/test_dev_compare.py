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
import sys
from pathlib import Path

import pytest
from pytest import CaptureFixture, MonkeyPatch

import nemo_gym.cli.main as cli_main
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


def write_jsonl(path: Path, rows: list[dict]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


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
                    "_ng_task_id": {"taskset": "tb", "task_id": "broken"},
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


class TestCliWiring:
    def test_positionals_and_flags_become_overrides(self, monkeypatch: MonkeyPatch) -> None:
        captured: dict = {}

        def fake_dispatch(target: str, overrides: list[str]) -> None:
            captured["target"], captured["overrides"] = target, overrides

        monkeypatch.setattr(cli_main, "dispatch", fake_dispatch)
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "gym",
                "dev",
                "compare",
                "old dir/old.jsonl",
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
        assert captured["target"] == "nemo_gym.cli.dev:dev_compare"
        assert captured["overrides"] == [
            '+old_rollouts="old dir/old.jsonl"',
            '+new_rollouts="new.jsonl"',
            '+key="verifier_metadata.task_id"',
            "+tail=3",
            '+logs_root="resources_servers/harbor"',
            '+json_output="out.json"',
        ]

    def test_optional_flags_default_to_nothing(self, monkeypatch: MonkeyPatch) -> None:
        captured: dict = {}
        monkeypatch.setattr(cli_main, "dispatch", lambda target, overrides: captured.update(overrides=overrides))
        monkeypatch.setattr(sys, "argv", ["gym", "dev", "compare", "a.jsonl", "b.jsonl"])
        main()
        assert captured["overrides"] == ['+old_rollouts="a.jsonl"', '+new_rollouts="b.jsonl"']

    def test_missing_positional_is_a_usage_error(self, monkeypatch: MonkeyPatch, capsys: CaptureFixture) -> None:
        monkeypatch.setattr(sys, "argv", ["gym", "dev", "compare", "only-one.jsonl"])
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2
        assert "NEW_ROLLOUTS" in capsys.readouterr().err

    def test_end_to_end_through_dispatch(
        self, monkeypatch: MonkeyPatch, tmp_path: Path, capsys: CaptureFixture
    ) -> None:
        # The real dispatch: argv is rewritten to Hydra overrides, the global config is built from them, and
        # `dev_compare` validates `RolloutCompareConfig` off it.
        old = write_jsonl(tmp_path / "old.jsonl", [old_row("a", 1.0), old_row("b", 1.0), old_row("c", 0.0)])
        new = write_jsonl(tmp_path / "new.jsonl", [new_row("a", 1.0), new_row("b", 0.0), new_row("c", 0.0)])
        json_path = tmp_path / "summary.json"
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
        assert json.loads(json_path.read_text())["counts"]["identical"] == 2
