# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import pytest

from nemo_gym.config_types import ConfigError
from nemo_gym.statistical_tests.common import (
    invoked_command,
    load_run_pair,
    report_stem,
    resolve_output_dir,
    sanitize_filename_part,
    write_reports,
)
from nemo_gym.statistical_tests.schema import STATS_SUBDIR_NAME, StatTestConfig
from tests.unit_tests.test_compare import _entry, _group, _write_run


AGENT = "bird_sql_simple_agent"
BASE = {
    "baseline_rollouts_jsonl_fpath": "runs/a/rollouts.jsonl",
    "candidate_rollouts_jsonl_fpaths": ["runs/b/r.jsonl"],
}


def _config(baseline, candidate, **overrides) -> StatTestConfig:
    return StatTestConfig.model_validate(
        {
            "baseline_rollouts_jsonl_fpath": str(baseline),
            "candidate_rollouts_jsonl_fpaths": [str(candidate)],
            **overrides,
        }
    )


class TestLoadRunPair:
    def test_loads_both_sides_and_narrows_to_the_sole_shared_agent(self, tmp_path):
        baseline = _write_run(tmp_path, "run_a", [_entry(groups=[_group(0, [1.0]), _group(1, [1.0])])])
        candidate = _write_run(tmp_path, "run_b", [_entry(groups=[_group(0, [0.0]), _group(1, [1.0])])])

        pair = load_run_pair(_config(baseline, candidate))

        assert pair.baseline_agent == AGENT and pair.candidate_agent == AGENT
        assert pair.baseline.num_tasks == 2 and pair.candidate.num_tasks == 2

    def test_an_ambiguous_agent_selection_is_an_error_not_a_loop(self, tmp_path):
        baseline = _write_run(
            tmp_path,
            "run_a",
            [_entry(agent="a1", groups=[_group(0, [1.0])]), _entry(agent="a2", groups=[_group(0, [1.0])])],
        )
        candidate = _write_run(
            tmp_path,
            "run_b",
            [_entry(agent="a1", groups=[_group(0, [0.0])]), _entry(agent="a2", groups=[_group(0, [0.0])])],
        )
        with pytest.raises(ConfigError, match="exactly one agent pair"):
            load_run_pair(_config(baseline, candidate))

    def test_an_explicit_agent_narrows_an_otherwise_ambiguous_pair(self, tmp_path):
        baseline = _write_run(
            tmp_path,
            "run_a",
            [_entry(agent="a1", groups=[_group(0, [1.0])]), _entry(agent="a2", groups=[_group(0, [1.0])])],
        )
        candidate = _write_run(
            tmp_path,
            "run_b",
            [_entry(agent="a1", groups=[_group(0, [0.0])]), _entry(agent="a2", groups=[_group(0, [0.0])])],
        )
        pair = load_run_pair(_config(baseline, candidate, agent_name="a2"))
        assert pair.baseline_agent == "a2" and pair.candidate_agent == "a2"

    def test_report_identity_describes_both_runs_and_the_selected_test(self, tmp_path):
        baseline = _write_run(tmp_path, "run_a", [_entry(groups=[_group(0, [1.0])])])
        candidate = _write_run(tmp_path, "run_b", [_entry(groups=[_group(0, [0.0])])])
        config = _config(baseline, candidate)

        identity = load_run_pair(config).report_identity(config, "gym eval stat-test ...")

        assert identity["test"] == "paired"
        assert identity["command"] == "gym eval stat-test ..."
        assert identity["baseline_task_count"] == 1 and identity["candidate_task_count"] == 1
        assert identity["generated_at"] and identity["nemo_gym_version"]


class TestReportStem:
    def test_leads_with_the_test_name_so_two_tests_cannot_overwrite_each_other(self):
        config = StatTestConfig.model_validate(BASE)
        assert report_stem(config) == "paired__alpha-0.05"

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [("reward", "reward"), ("a/b", "a-b"), ("pass@1[avg-of-2]", "pass-1-avg-of-2"), ("--x--", "x")],
    )
    def test_sanitize_makes_a_filename_safe_token(self, raw, expected):
        assert sanitize_filename_part(raw) == expected


class TestResolveOutputDir:
    def test_unset_nests_under_the_candidate_runs_own_directory(self, tmp_path):
        config = StatTestConfig.model_validate(
            {**BASE, "candidate_rollouts_jsonl_fpaths": [str(tmp_path / "run_b" / "rollouts.jsonl")]}
        )
        assert resolve_output_dir(config) == tmp_path / "run_b" / STATS_SUBDIR_NAME

    def test_explicit_path_is_used_literally_with_no_nesting(self, tmp_path):
        config = StatTestConfig.model_validate({**BASE, "output_dirpath": str(tmp_path / "elsewhere")})
        assert resolve_output_dir(config) == tmp_path / "elsewhere"


class TestWriteReports:
    def _write(self, output_dir, report_format="both"):
        return write_reports(
            output_dir, "stem", report_format=report_format, markdown="plain text", payload={"schema_version": "1"}
        )

    @pytest.mark.parametrize(
        ("report_format", "expected"),
        [("both", ["stem.md", "stem.json"]), ("md", ["stem.md"]), ("json", ["stem.json"])],
    )
    def test_report_format_selects_the_artifacts(self, tmp_path, report_format, expected):
        written = self._write(tmp_path, report_format)
        assert [path.name for path in written] == expected
        assert all(path.exists() for path in written)

    def test_an_output_dir_that_is_a_file_is_rejected_cleanly(self, tmp_path):
        not_a_dir = tmp_path / "file"
        not_a_dir.write_text("")
        with pytest.raises(ConfigError, match="exists and is not a directory"):
            self._write(not_a_dir)


class TestInvokedCommand:
    def test_records_the_subcommand_that_actually_ran(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["gym", "+no_stats=false"])
        assert invoked_command() == "gym eval stat-test +no_stats=false"
        assert invoked_command("compare") == "gym eval compare +no_stats=false"

    def test_quotes_awkward_overrides_and_redacts_secrets(self, monkeypatch):
        monkeypatch.setattr("sys.argv", ["gym", "+baseline_rollouts_jsonl_fpath=a b.jsonl"])
        assert invoked_command() == "gym eval stat-test '+baseline_rollouts_jsonl_fpath=a b.jsonl'"
