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
from nemo_gym.statistical_tests.schema import STATS_SUBDIR_NAME, StatTestConfig, StatTestReport
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

        assert identity["test"] == "paired-t-test"
        assert identity["command"] == "gym eval stat-test ..."
        assert identity["baseline_task_count"] == 1 and identity["candidate_task_count"] == 1
        assert identity["generated_at"] and identity["nemo_gym_version"]


class TestReportStem:
    def _report(self, **kw) -> StatTestReport:
        fields = {
            "generated_at": "2026-01-01T00:00:00+00:00",
            "nemo_gym_version": "0.0.0",
            "command": "gym eval stat-test ...",
            "test": "paired-t-test",
            "baseline_rollouts_jsonl_fpath": "runs/run_a/rollouts.jsonl",
            "baseline_aggregate_metrics_fpath": "runs/run_a/rollouts_aggregate_metrics.json",
            "candidate_rollouts_jsonl_fpath": "runs/run_b/rollouts.jsonl",
            "candidate_aggregate_metrics_fpath": "runs/run_b/rollouts_aggregate_metrics.json",
            "baseline_agent": "default",
            "candidate_agent": "default",
            "baseline_task_count": 4,
            "candidate_task_count": 4,
        }
        return StatTestReport(**{**fields, **kw})

    def test_leads_with_the_test_name_so_two_tests_cannot_overwrite_each_other(self):
        config = StatTestConfig.model_validate(BASE)
        stem = report_stem(config, self._report())
        assert stem.startswith("paired-t-test__")
        assert stem.startswith("paired-t-test__run_a-rollouts__run_b-rollouts__agent-default__alpha-0.05__")

    def test_two_runs_sharing_a_leaf_directory_name_do_not_overwrite_each_other(self):
        """Only the leaf directory is readable in the stem, so `alpha/run_b` and `beta/run_b` collide there."""
        config = StatTestConfig.model_validate({**BASE, "output_dirpath": "/tmp/reports"})
        from_alpha = report_stem(config, self._report(candidate_rollouts_jsonl_fpath="alpha/run_b/rollouts.jsonl"))
        from_beta = report_stem(config, self._report(candidate_rollouts_jsonl_fpath="beta/run_b/rollouts.jsonl"))
        assert from_alpha != from_beta
        assert from_alpha.rsplit("__", 1)[0] == from_beta.rsplit("__", 1)[0]  # identical but for the digest

    def test_a_second_baseline_against_the_same_candidate_does_not_overwrite_the_first(self):
        """The default output dir is the candidate's own, so only the stem separates two baselines."""
        config = StatTestConfig.model_validate(BASE)
        against_a = report_stem(config, self._report())
        against_c = report_stem(config, self._report(baseline_rollouts_jsonl_fpath="runs/run_c/rollouts.jsonl"))
        assert against_a != against_c

    def test_a_second_candidate_under_one_output_dir_does_not_overwrite_the_first(self):
        config = StatTestConfig.model_validate({**BASE, "output_dirpath": "/tmp/reports"})
        into_b = report_stem(config, self._report())
        into_d = report_stem(config, self._report(candidate_rollouts_jsonl_fpath="runs/run_d/rollouts.jsonl"))
        assert into_b != into_d

    def test_a_second_agent_on_the_same_run_pair_does_not_overwrite_the_first(self):
        config = StatTestConfig.model_validate(BASE)
        first = report_stem(config, self._report(baseline_agent="a1", candidate_agent="a1"))
        second = report_stem(config, self._report(baseline_agent="a2", candidate_agent="a2"))
        assert first != second
        assert "agent-a1" in first and "agent-a2" in second

    def test_a_cross_agent_pair_names_both_sides_in_baseline_then_candidate_order(self):
        config = StatTestConfig.model_validate(BASE)
        stem = report_stem(config, self._report(baseline_agent="a1", candidate_agent="a2"))
        assert "agent-a1-vs-a2" in stem

    def test_rerunning_the_same_invocation_reuses_the_same_filename(self):
        config = StatTestConfig.model_validate(BASE)
        assert report_stem(config, self._report()) == report_stem(config, self._report())

    @pytest.mark.parametrize(
        ("fpath", "expected"),
        [
            ("runs/run_a/rollouts.jsonl", "run_a-rollouts"),
            ("a.jsonl", "a"),
            ("runs/aime24_bf16.jsonl", "runs-aime24_bf16"),
        ],
    )
    def test_each_run_is_labelled_by_its_directory_and_its_stem(self, fpath, expected):
        """A flat layout has no per-run directory and a nested one always names the file `rollouts`,
        so only the pair of them tells two runs apart."""
        stem = report_stem(StatTestConfig.model_validate(BASE), self._report(baseline_rollouts_jsonl_fpath=fpath))
        assert stem.split("__")[1] == expected

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

    def test_an_explicit_path_still_nests_the_statistical_tests_directory_inside_it(self, tmp_path):
        config = StatTestConfig.model_validate({**BASE, "output_dirpath": str(tmp_path / "elsewhere")})
        assert resolve_output_dir(config) == tmp_path / "elsewhere" / STATS_SUBDIR_NAME


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
