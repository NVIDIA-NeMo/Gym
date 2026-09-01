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

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import orjson
import pytest

from nemo_gym.comparison.diff import build_flip_summary, build_metric_rows, compare_runs, is_comparable_metric
from nemo_gym.comparison.loading import (
    build_loaded_run,
    load_agg_metrics_file,
    resolve_agent_selections,
)
from nemo_gym.comparison.report import render_markdown, summary_lines, write_reports
from nemo_gym.comparison.runner import build_comparison_result, resolve_output_dir
from nemo_gym.comparison.schema import ComparisonConfig
from nemo_gym.config_types import ConfigError, ConfigPathNotFoundError
from nemo_gym.path_utils import aggregate_metrics_path_for


AGENT = "bird_sql_simple_agent"


def _group(
    task_index: int,
    rewards: List[float],
    *,
    extra: Optional[Dict[str, Any]] = None,
    with_rollout_infos: bool = True,
) -> Dict[str, Any]:
    """One `group_level_metrics` entry, shaped like what aggregation really writes."""
    group: Dict[str, Any] = {
        "_ng_task_index": task_index,
        "mean/reward": sum(rewards) / len(rewards),
        "min/reward": min(rewards),
        "max/reward": max(rewards),
        "num_rollouts": len(rewards),
        "expected_num_rollouts": len(rewards),
        "missing_num_rollouts": 0,
    }
    if with_rollout_infos:
        group["rollout_infos"] = [
            {
                "rollout_id": f"{task_index}:{rollout_index}",
                "_ng_task_index": task_index,
                "_ng_rollout_index": rollout_index,
                "reward": reward,
            }
            for rollout_index, reward in enumerate(rewards)
        ]
    group.update(extra or {})
    return group


def _entry(
    *,
    agent: str = AGENT,
    agent_metrics: Optional[Dict[str, Any]] = None,
    key_metrics: Optional[Dict[str, Any]] = None,
    groups: Optional[List[Dict[str, Any]]] = None,
    repeat_level_metrics: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "agent_ref": {"name": agent},
        "agent_metrics": agent_metrics if agent_metrics is not None else {"mean/reward": 0.5},
        "key_metrics": key_metrics if key_metrics is not None else {"mean/reward": 0.5},
        "group_level_metrics": groups if groups is not None else [],
        "repeat_level_metrics": repeat_level_metrics if repeat_level_metrics is not None else [],
    }


def _write_run(tmp_path: Path, name: str, entries: List[Dict[str, Any]], *, write_rollouts: bool = True) -> Path:
    run_dir = tmp_path / name
    run_dir.mkdir(parents=True, exist_ok=True)
    rollouts = run_dir / "rollouts.jsonl"
    if write_rollouts:
        rollouts.write_text("")
    aggregate_metrics_path_for(rollouts).write_bytes(orjson.dumps(entries))
    return rollouts


def _load(tmp_path: Path, name: str, entries: List[Dict[str, Any]], *, role="baseline", agent: str = AGENT):
    rollouts = _write_run(tmp_path, name, entries)
    run_file = load_agg_metrics_file(str(rollouts), role=role)
    return build_loaded_run(run_file, agent)


class TestAggregateMetricsPath:
    def test_derives_sibling_from_rollouts_path(self):
        assert aggregate_metrics_path_for(Path("results/rollouts.jsonl")) == Path(
            "results/rollouts_aggregate_metrics.json"
        )

    def test_dotted_stem_matches_the_writer(self):
        # with_stem/with_suffix (not with_name) is what rollout collection uses, so a dotted stem
        # resolves to the same file the writer produced.
        assert aggregate_metrics_path_for(Path("out/rollouts.v2.jsonl")) == Path(
            "out/rollouts.v2_aggregate_metrics.json"
        )


class TestLoadRunFile:
    def test_reads_entries_keyed_by_agent(self, tmp_path):
        rollouts = _write_run(tmp_path, "base", [_entry()])
        run_file = load_agg_metrics_file(str(rollouts), role="baseline")
        assert run_file.agent_names == [AGENT]
        assert run_file.aggregate_metrics_fpath == aggregate_metrics_path_for(rollouts)

    def test_missing_rollouts_and_metrics_names_the_run(self, tmp_path):
        with pytest.raises(ConfigPathNotFoundError, match="Aggregate metrics not found"):
            load_agg_metrics_file(str(tmp_path / "nope" / "rollouts.jsonl"), role="baseline")

    def test_missing_metrics_with_present_rollouts_explains_disable_aggregation(self, tmp_path):
        run_dir = tmp_path / "partial"
        run_dir.mkdir()
        (run_dir / "rollouts.jsonl").write_text("")
        with pytest.raises(ConfigPathNotFoundError, match="disable-aggregation"):
            load_agg_metrics_file(str(run_dir / "rollouts.jsonl"), role="baseline")

    def test_override_is_used_instead_of_the_sibling(self, tmp_path):
        rollouts = _write_run(tmp_path, "base", [_entry()])
        override = tmp_path / "elsewhere.json"
        override.write_bytes(orjson.dumps([_entry(agent="other")]))
        run_file = load_agg_metrics_file(
            str(rollouts),
            role="baseline",
            aggregate_metrics_fpath_override=str(override),
        )
        assert run_file.agent_names == ["other"]

    @pytest.mark.parametrize(
        "payload, message",
        [
            ({"agent_ref": {"name": AGENT}}, "expected a JSON list"),
            ([], "no agent entries"),
            ([[1, 2]], "is not an object"),
            ([{"agent_metrics": {}}], "has no `agent_ref.name`"),
            ([_entry(), _entry()], "two entries for agent"),
        ],
    )
    def test_malformed_files_are_rejected(self, tmp_path, payload, message):
        run_dir = tmp_path / "bad"
        run_dir.mkdir()
        rollouts = run_dir / "rollouts.jsonl"
        rollouts.write_text("")
        aggregate_metrics_path_for(rollouts).write_bytes(orjson.dumps(payload))
        with pytest.raises(ConfigError, match=message):
            load_agg_metrics_file(str(rollouts), role="baseline")

    def test_a_directory_in_place_of_the_metrics_file_is_rejected_cleanly(self, tmp_path):
        """`exists()` is true for a directory, so the read itself has to fail cleanly."""
        rollouts = _write_run(tmp_path, "base", [_entry()])
        with pytest.raises(ConfigError, match="Cannot read aggregate metrics"):
            load_agg_metrics_file(
                str(rollouts),
                role="baseline",
                aggregate_metrics_fpath_override=str(tmp_path / "base"),
            )

    def test_an_unreadable_metrics_file_is_rejected_cleanly(self, tmp_path):
        rollouts = _write_run(tmp_path, "base", [_entry()])
        metrics = aggregate_metrics_path_for(rollouts)
        metrics.chmod(0o000)
        try:
            if os.access(metrics, os.R_OK):
                pytest.skip("cannot make a file unreadable as this user (running as root?)")
            with pytest.raises(ConfigError, match="Cannot read aggregate metrics"):
                load_agg_metrics_file(str(rollouts), role="baseline")
        finally:
            metrics.chmod(0o644)

    def test_invalid_json_is_rejected(self, tmp_path):
        run_dir = tmp_path / "bad"
        run_dir.mkdir()
        rollouts = run_dir / "rollouts.jsonl"
        rollouts.write_text("")
        aggregate_metrics_path_for(rollouts).write_text("{not json")
        with pytest.raises(ConfigError, match="is not valid JSON"):
            load_agg_metrics_file(str(rollouts), role="baseline")


class TestNumRepeatsDerivation:
    def test_prefers_expected_num_rollouts(self, tmp_path):
        run = _load(tmp_path, "base", [_entry(groups=[_group(0, [1.0, 0.0, 1.0])])])
        assert run.num_repeats == 3

    def test_falls_back_to_repeat_level_metrics(self, tmp_path):
        groups = [{"_ng_task_index": 0, "mean/reward": 1.0}]
        entry = _entry(groups=groups, repeat_level_metrics=[{"_ng_rollout_index": i} for i in range(4)])
        run = _load(tmp_path, "base", [entry])
        assert run.num_repeats == 4

    def test_repeat_level_fallback_works_per_agent_in_a_multi_agent_file(self, tmp_path):
        """Aggregation nests each agent's repeat_level_metrics under its own entry, stripped of
        `agent_ref`, so the fallback is per-agent regardless of how many agents the file holds."""
        groups = [{"_ng_task_index": 0, "mean/reward": 1.0}]
        entries = [
            _entry(agent="a", groups=groups, repeat_level_metrics=[{"_ng_rollout_index": i} for i in range(2)]),
            _entry(agent="b", groups=groups, repeat_level_metrics=[{"_ng_rollout_index": i} for i in range(7)]),
        ]
        assert _load(tmp_path, "multi", entries, agent="a").num_repeats == 2
        assert _load(tmp_path, "multi", entries, agent="b").num_repeats == 7

    def test_a_partially_recovered_run_reports_its_full_repeat_count(self, tmp_path):
        """Some tasks come up short when a run is partially recovered; the run still had 3 repeats.

        Taking the mode instead would report 1 here, and would depend on task ordering when the
        per-task counts tie.
        """
        groups = [_group(0, [1.0, 0.0, 1.0]), _group(1, [1.0]), _group(2, [0.0, 1.0])]
        run = _load(tmp_path, "base", [_entry(groups=groups)])
        assert run.num_repeats == 3

    def test_unknown_when_nothing_records_it(self, tmp_path):
        run = _load(tmp_path, "base", [_entry(groups=[{"_ng_task_index": 0, "mean/reward": 1.0}])])
        assert run.num_repeats is None

    def test_repeat_cis_flag_tracks_the_ci_keys(self, tmp_path):
        without = _load(tmp_path, "a", [_entry(agent_metrics={"mean/reward": 0.5})])
        with_ci = _load(
            tmp_path,
            "b",
            [_entry(agent_metrics={"mean/reward": 0.5, "ci_low_95_across_repeats/mean/reward": 0.4})],
        )
        assert not without.has_repeat_cis
        assert with_ci.has_repeat_cis


class TestAgentSelection:
    def _files(self, tmp_path, baseline_agents, candidate_agents):
        baseline = load_agg_metrics_file(
            str(_write_run(tmp_path, "base", [_entry(agent=a) for a in baseline_agents])),
            role="baseline",
        )
        candidate = load_agg_metrics_file(
            str(_write_run(tmp_path, "cand", [_entry(agent=a) for a in candidate_agents])),
            role="candidate",
        )
        return baseline, candidate

    def test_full_join_compares_every_shared_agent(self, tmp_path):
        baseline, candidate = self._files(tmp_path, ["a", "b"], ["a", "b"])
        selections, warnings, skipped = resolve_agent_selections(baseline, [candidate])
        assert [s.baseline_agent for s in selections] == ["a", "b"]
        assert warnings == [] and skipped == {}

    def test_full_join_warns_about_agents_only_one_side_has(self, tmp_path):
        baseline, candidate = self._files(tmp_path, ["a", "b"], ["a"])
        selections, warnings, skipped = resolve_agent_selections(baseline, [candidate])
        assert [s.baseline_agent for s in selections] == ["a"]
        assert skipped == {"baseline": ["b"]}
        assert "b" in warnings[0]

    def test_disjoint_agent_names_is_fatal(self, tmp_path):
        baseline, candidate = self._files(tmp_path, ["a"], ["b"])
        with pytest.raises(ConfigError, match="No agent name is present in every run"):
            resolve_agent_selections(baseline, [candidate])

    def test_agent_flag_selects_one_agent_on_both_sides(self, tmp_path):
        baseline, candidate = self._files(tmp_path, ["a", "b"], ["a", "b"])
        selections, _, _ = resolve_agent_selections(baseline, [candidate], agent_name="b")
        assert selections == [selections[0]]
        assert selections[0].baseline_agent == "b" and selections[0].candidate_agents == ("b",)

    def test_unknown_agent_name_is_fatal_with_a_suggestion(self, tmp_path):
        baseline, candidate = self._files(tmp_path, ["alpha"], ["alpha"])
        with pytest.raises(ConfigError, match="Did you mean `alpha`"):
            resolve_agent_selections(baseline, [candidate], agent_name="alpah")

    def test_per_side_agents_may_differ(self, tmp_path):
        baseline, candidate = self._files(tmp_path, ["old_name"], ["new_name"])
        selections, _, _ = resolve_agent_selections(
            baseline, [candidate], baseline_agent_name="old_name", candidate_agent_names=["new_name"]
        )
        assert selections[0].baseline_agent == "old_name"
        assert selections[0].candidate_agents == ("new_name",)

    def test_per_side_selection_falls_back_to_a_sole_agent(self, tmp_path):
        baseline, candidate = self._files(tmp_path, ["only"], ["new_name"])
        selections, _, _ = resolve_agent_selections(baseline, [candidate], candidate_agent_names=["new_name"])
        assert selections[0].baseline_agent == "only"

    def test_per_side_selection_needs_a_flag_when_a_side_is_ambiguous(self, tmp_path):
        baseline, candidate = self._files(tmp_path, ["a", "b"], ["new_name"])
        with pytest.raises(ConfigError, match="contain 2 agents: a, b"):
            resolve_agent_selections(baseline, [candidate], candidate_agent_names=["new_name"])


class TestMetricRows:
    @pytest.mark.parametrize(
        "name, comparable",
        [
            ("mean/reward", True),
            ("pass@1[avg-of-3]/accuracy", True),
            ("simple/pass@3/accuracy", True),
            ("std/reward", False),
            ("median/reward", False),
            ("ci_low_95/reward", False),
            ("mean_across_repeats/mean/reward", False),
            ("ci_high_95_across_repeats/mean/reward", False),
            ("pass@1[avg-of-3]/accuracy/std_err_across_runs", False),
            ("pass@1[avg-of-3]/accuracy/avg_sample_std_dev", False),
        ],
    )
    def test_only_real_metrics_get_a_row(self, name, comparable):
        assert is_comparable_metric(name) is comparable

    def test_reads_values_ci_and_delta(self, tmp_path):
        baseline = _load(
            tmp_path,
            "base",
            [
                _entry(
                    agent_metrics={
                        "mean/reward": 0.80,
                        "ci_low_95_across_repeats/mean/reward": 0.70,
                        "ci_high_95_across_repeats/mean/reward": 0.90,
                        "se_across_repeats/mean/reward": 0.05,
                        "mean_across_repeats/mean/reward": 0.80,
                    },
                    key_metrics={"mean/reward": 0.80},
                )
            ],
        )
        candidate = _load(
            tmp_path,
            "cand",
            [_entry(agent_metrics={"mean/reward": 0.60}, key_metrics={"mean/reward": 0.60})],
            role="candidate",
        )
        (row,) = build_metric_rows(baseline, [candidate])
        assert row.metric == "mean/reward" and row.is_key_metric
        assert row.baseline.value == pytest.approx(0.80)
        assert (row.baseline.ci_low, row.baseline.ci_high) == (0.70, 0.90)
        assert row.baseline.se_across_repeats == pytest.approx(0.05)
        assert row.candidates[0].delta == pytest.approx(-0.20)
        assert row.candidates[0].delta_pct == pytest.approx(-25.0)
        # The candidate recorded no interval of its own.
        assert row.candidates[0].ci_low is None

    def test_zero_baseline_leaves_relative_change_undefined(self, tmp_path):
        baseline = _load(tmp_path, "base", [_entry(agent_metrics={"mean/reward": 0.0})])
        candidate = _load(tmp_path, "cand", [_entry(agent_metrics={"mean/reward": 0.5})], role="candidate")
        (row,) = build_metric_rows(baseline, [candidate])
        assert row.candidates[0].delta == pytest.approx(0.5)
        assert row.candidates[0].delta_pct is None

    def test_one_sided_and_non_numeric_metrics(self, tmp_path):
        baseline = _load(
            tmp_path,
            "base",
            [_entry(agent_metrics={"mean/reward": 0.5, "pass@1/accuracy": 40.0, "per_sample_aggregate": {"a": [1]}})],
        )
        candidate = _load(
            tmp_path,
            "cand",
            [_entry(agent_metrics={"mean/reward": 0.5, "pass@2/accuracy": 55.0})],
            role="candidate",
        )
        rows = {row.metric: row for row in build_metric_rows(baseline, [candidate])}
        assert "per_sample_aggregate" not in rows
        assert rows["pass@1/accuracy"].present_in == ["baseline"]
        assert rows["pass@1/accuracy"].candidates == [None]
        assert rows["pass@2/accuracy"].present_in == ["candidate[0]"]
        assert rows["mean/reward"].present_in == ["baseline", "candidate[0]"]

    def test_key_metrics_only_name_still_gets_a_row(self, tmp_path):
        """key_metrics can rename or synthesize a name that never appears in agent_metrics (e.g. an
        ASR server renaming corpus_wer@k=N -> wer). That name must still earn a row, flagged as a
        key metric, not silently vanish."""
        baseline = _load(
            tmp_path,
            "base",
            [_entry(agent_metrics={"corpus_wer@k=4": 12.5}, key_metrics={"wer": 12.5})],
        )
        candidate = _load(
            tmp_path,
            "cand",
            [_entry(agent_metrics={"corpus_wer@k=4": 10.0}, key_metrics={"wer": 10.0})],
            role="candidate",
        )
        rows = {row.metric: row for row in build_metric_rows(baseline, [candidate])}
        assert "wer" in rows
        assert rows["wer"].is_key_metric
        assert rows["wer"].baseline.value == pytest.approx(12.5)
        assert rows["wer"].candidates[0].value == pytest.approx(10.0)
        assert rows["wer"].candidates[0].delta == pytest.approx(-2.5)

    def test_key_metrics_name_shared_with_agent_metrics_prefers_the_agent_metrics_value(self, tmp_path):
        """On a name collision, the agent_metrics value wins (it's the one CI/std-err lookups key
        off of); key_metrics only fills in names agent_metrics doesn't already have."""
        baseline = _load(
            tmp_path,
            "base",
            [_entry(agent_metrics={"mean/reward": 0.5}, key_metrics={"mean/reward": 0.999})],
        )
        (row,) = build_metric_rows(baseline, [])
        assert row.baseline.value == pytest.approx(0.5)


class TestFlips:
    def test_binary_mode_counts_flips_in_both_directions(self, tmp_path):
        baseline = _load(
            tmp_path,
            "base",
            [_entry(groups=[_group(0, [1.0, 1.0]), _group(1, [0.0, 0.0]), _group(2, [1.0, 1.0])])],
        )
        candidate = _load(
            tmp_path,
            "cand",
            [_entry(groups=[_group(0, [0.0, 0.0]), _group(1, [1.0, 1.0]), _group(2, [1.0, 1.0])])],
            role="candidate",
        )
        summary = build_flip_summary(baseline, candidate)
        assert summary.mode == "binary"
        assert summary.field == "reward"
        assert (summary.pass_to_fail_count, summary.fail_to_pass_count) == (1, 1)
        assert summary.unchanged_count == 1 and summary.net == 0
        # Regressions are listed first.
        assert [flip.direction for flip in summary.flips] == ["pass_to_fail", "fail_to_pass"]
        assert summary.flips[0].baseline_rewards == [1.0, 1.0]
        assert summary.flips[0].candidate_rewards == [0.0, 0.0]

    def test_an_evenly_split_task_is_tied_not_flipped(self, tmp_path):
        baseline = _load(tmp_path, "base", [_entry(groups=[_group(0, [1.0, 0.0])])])
        candidate = _load(tmp_path, "cand", [_entry(groups=[_group(0, [1.0, 1.0])])], role="candidate")
        summary = build_flip_summary(baseline, candidate)
        assert summary.tied_count == 1
        assert summary.flips == []

    def test_continuous_rewards_report_largest_movers(self, tmp_path):
        baseline = _load(tmp_path, "base", [_entry(groups=[_group(0, [0.25]), _group(1, [0.80])])])
        candidate = _load(
            tmp_path,
            "cand",
            [_entry(groups=[_group(0, [0.30]), _group(1, [0.10])])],
            role="candidate",
        )
        summary = build_flip_summary(baseline, candidate)
        assert summary.mode == "continuous"
        assert summary.pass_to_fail_count is None
        assert [flip.task_index for flip in summary.flips] == [1, 0]
        assert all(flip.direction == "changed" for flip in summary.flips)

    def test_no_overlapping_tasks_is_reported_not_raised(self, tmp_path):
        baseline = _load(tmp_path, "base", [_entry(groups=[_group(0, [1.0])])])
        candidate = _load(tmp_path, "cand", [_entry(groups=[_group(7, [1.0])])], role="candidate")
        summary = build_flip_summary(baseline, candidate)
        assert summary.mode == "unavailable"
        assert "no overlapping task indices" in summary.reason

    def test_missing_per_task_metrics_is_reported(self, tmp_path):
        baseline = _load(tmp_path, "base", [_entry(groups=[])])
        candidate = _load(tmp_path, "cand", [_entry(groups=[_group(0, [1.0])])], role="candidate")
        assert build_flip_summary(baseline, candidate).mode == "unavailable"

    def test_missing_reward_field_is_reported(self, tmp_path):
        groups = [{"_ng_task_index": 0, "mean/score": 1.0}]
        baseline = _load(tmp_path, "base", [_entry(groups=groups)])
        candidate = _load(tmp_path, "cand", [_entry(groups=groups)], role="candidate")
        summary = build_flip_summary(baseline, candidate)
        assert summary.mode == "unavailable"
        assert "mean/reward" in summary.reason

    def test_continuous_mode_only_keeps_reward_lists_for_shown_flips(self, tmp_path):
        from nemo_gym.comparison.schema import MAX_FLIPS_SHOWN

        num_tasks = MAX_FLIPS_SHOWN + 1
        baseline = _load(
            tmp_path,
            "base",
            [_entry(groups=[_group(i, [0.5]) for i in range(num_tasks)])],
        )
        candidate = _load(
            tmp_path,
            "cand",
            # Distinct deltas so the sort order (largest |delta| first) is deterministic.
            [_entry(groups=[_group(i, [0.5 + 0.01 * (num_tasks - i)]) for i in range(num_tasks)])],
            role="candidate",
        )
        summary = build_flip_summary(baseline, candidate)
        assert summary.mode == "continuous"
        assert len(summary.flips) == num_tasks
        with_rewards = [flip for flip in summary.flips if flip.baseline_rewards is not None]
        without_rewards = [flip for flip in summary.flips if flip.baseline_rewards is None]
        assert len(with_rewards) == MAX_FLIPS_SHOWN
        assert len(without_rewards) == 1
        # The dropped flip is the smallest mover, not an arbitrary one.
        assert without_rewards[0] is summary.flips[-1]

    def test_rollout_infos_absent_leaves_per_repeat_rewards_empty(self, tmp_path):
        baseline = _load(tmp_path, "base", [_entry(groups=[_group(0, [1.0], with_rollout_infos=False)])])
        candidate = _load(
            tmp_path,
            "cand",
            [_entry(groups=[_group(0, [0.0], with_rollout_infos=False)])],
            role="candidate",
        )
        summary = build_flip_summary(baseline, candidate)
        assert summary.flips[0].baseline_rewards is None


class TestCompareRuns:
    def test_no_shared_metrics_is_fatal(self, tmp_path):
        # key_metrics must be disjoint too (not just agent_metrics): build_metric_rows folds
        # key_metrics into the compared metrics, so leaving them at _entry()'s shared default would
        # give the two runs an overlapping "mean/reward" key and this wouldn't be fatal anymore.
        baseline = _load(
            tmp_path,
            "base",
            [_entry(agent_metrics={"pass@1[avg-of-3]/accuracy": 40.0}, key_metrics={})],
        )
        candidate = _load(
            tmp_path,
            "cand",
            [_entry(agent_metrics={"pass@1[avg-of-5]/accuracy": 44.0}, key_metrics={})],
            role="candidate",
        )
        with pytest.raises(ConfigError, match="share no metric keys"):
            compare_runs(baseline, [candidate])

    def test_notes_flag_repeat_mismatch_and_absent_intervals(self, tmp_path):
        baseline = _load(tmp_path, "base", [_entry(groups=[_group(0, [1.0, 1.0, 1.0])])])
        candidate = _load(tmp_path, "cand", [_entry(groups=[_group(0, [1.0, 0.0])])], role="candidate")
        comparison = compare_runs(baseline, [candidate])
        assert comparison.baseline_repeat_count == 3 and comparison.candidate_repeat_counts == [2]
        assert any("Repeat counts differ" in note for note in comparison.notes)
        assert any("confidence intervals" in note for note in comparison.notes)


class TestRunLabel:
    def test_label_is_the_run_directory(self, tmp_path):
        run_file = load_agg_metrics_file(str(_write_run(tmp_path, "run_a", [_entry()])), role="baseline")
        assert run_file.label == "run_a"

    def test_label_falls_back_to_the_stem_when_there_is_no_directory(self):
        from nemo_gym.comparison.schema import RunFile

        run_file = RunFile(
            role="baseline",
            rollouts_jsonl_fpath=Path("rollouts.jsonl"),
            aggregate_metrics_fpath=Path("rollouts_aggregate_metrics.json"),
        )
        assert run_file.label == "rollouts"


class TestComparisonConfig:
    def _config(self, **overrides) -> Dict[str, Any]:
        return {
            "baseline_rollouts_jsonl_fpath": "a/rollouts.jsonl",
            "candidate_rollouts_jsonl_fpaths": ["b/rollouts.jsonl"],
            **overrides,
        }

    def test_single_candidate_validates(self):
        config = ComparisonConfig.model_validate(self._config())
        assert config.report_format == "both"
        assert config.output_dirpath is None

    def test_multiple_candidates_are_rejected_for_now(self):
        payload = self._config(candidate_rollouts_jsonl_fpaths=["b/r.jsonl", "c/r.jsonl"])
        with pytest.raises(ValueError, match="is not supported yet"):
            ComparisonConfig.model_validate(payload)

    def test_candidate_agent_list_must_match_candidate_count(self):
        payload = self._config(candidate_agent_names=["one", "two"])
        with pytest.raises(ValueError, match="candidate_agent_names has 2 entries"):
            ComparisonConfig.model_validate(payload)

    def test_metrics_override_list_must_match_candidate_count(self):
        payload = self._config(candidate_aggregate_metrics_fpaths=["x.json", "y.json"])
        with pytest.raises(ValueError, match="candidate_aggregate_metrics_fpaths has 2 entries"):
            ComparisonConfig.model_validate(payload)

    @pytest.mark.parametrize(
        "payload_kwargs, field_name",
        [
            ({"candidate_agent_names": ["one", "two"]}, "candidate_agent_names"),
            ({"candidate_aggregate_metrics_fpaths": ["x.json", "y.json"]}, "candidate_aggregate_metrics_fpaths"),
        ],
    )
    def test_every_per_candidate_list_reports_the_same_way(self, payload_kwargs, field_name):
        """Both per-candidate lists name their own field and say what to do, in one shared wording."""
        with pytest.raises(ValueError, match=f"{field_name} has 2 entries but 1 candidate"):
            ComparisonConfig.model_validate(self._config(**payload_kwargs))
        with pytest.raises(ValueError, match="Give one entry per candidate, in the same order"):
            ComparisonConfig.model_validate(self._config(**payload_kwargs))


class TestCliFlagTranslation:
    """`gym compare`'s flags must survive the argparse -> Hydra override -> pydantic round trip."""

    def _overrides(self, argv: List[str]) -> List[str]:
        from nemo_gym.cli.main import build_parser

        args, unknown = build_parser().parse_known_args(argv)
        assert unknown == [], f"flags left unparsed: {unknown}"
        return [token for flag in args._command.flags for token in flag.translate_to_hydra(args)]

    def _config(self, argv: List[str]) -> ComparisonConfig:
        from hydra.core.override_parser.overrides_parser import OverridesParser

        parsed = OverridesParser.create().parse_overrides(self._overrides(argv))
        return ComparisonConfig.model_validate({o.key_or_group: o.value() for o in parsed})

    def test_paths_round_trip_through_hydra(self, tmp_path):
        config = self._config(
            ["eval", "compare", "--baseline", "runs/a/rollouts.jsonl", "--candidates", "runs/b/rollouts.jsonl"]
        )
        assert config.baseline_rollouts_jsonl_fpath == "runs/a/rollouts.jsonl"
        assert config.candidate_rollouts_jsonl_fpaths == ["runs/b/rollouts.jsonl"]

    @pytest.mark.parametrize(
        "path",
        [
            "runs/with space/rollouts.jsonl",
            "runs/v=2/rollouts.jsonl",
            "runs/[v2]/rollouts.jsonl",
        ],
    )
    def test_awkward_paths_survive_verbatim(self, path):
        config = self._config(["eval", "compare", "--baseline", path, "--candidates", path, "--output-dir", path])
        assert config.baseline_rollouts_jsonl_fpath == path
        assert config.candidate_rollouts_jsonl_fpaths == [path]
        assert config.output_dirpath == path

    def test_comma_separated_candidates_split_into_a_list(self):
        overrides = self._overrides(["eval", "compare", "--baseline", "a.jsonl", "--candidates", "b.jsonl, c.jsonl"])
        assert '+candidate_rollouts_jsonl_fpaths=["b.jsonl","c.jsonl"]' in overrides

    def test_report_format_and_agent_flags_translate(self):
        overrides = self._overrides(
            [
                "eval",
                "compare",
                "--baseline",
                "a.jsonl",
                "--candidates",
                "b.jsonl",
                "--report-format",
                "md",
                "--agent",
                "my_agent",
                "--candidate-agents",
                "other_agent",
            ]
        )
        assert "+report_format=md" in overrides
        assert "+agent_name=my_agent" in overrides
        assert '+candidate_agent_names=["other_agent"]' in overrides

    def test_agg_metrics_override_flags_translate(self):
        config = self._config(
            [
                "eval",
                "compare",
                "--baseline",
                "a.jsonl",
                "--candidates",
                "b.jsonl",
                "--baseline-agg-metrics",
                "elsewhere/base.json",
                "--candidates-agg-metrics",
                "elsewhere/cand.json",
            ]
        )
        assert config.baseline_aggregate_metrics_fpath == "elsewhere/base.json"
        assert config.candidate_aggregate_metrics_fpaths == ["elsewhere/cand.json"]

    def test_unset_flags_contribute_no_overrides(self):
        overrides = self._overrides(["eval", "compare", "--baseline", "a.jsonl", "--candidates", "b.jsonl"])
        assert not [
            token
            for token in overrides
            if "output_dirpath" in token or "agent" in token or "aggregate_metrics" in token
        ]


class TestEndToEnd:
    def _result(self, tmp_path, **config_overrides):
        baseline_rollouts = _write_run(
            tmp_path,
            "run_a",
            [
                _entry(
                    agent_metrics={
                        "mean/reward": 0.75,
                        "ci_low_95_across_repeats/mean/reward": 0.70,
                        "ci_high_95_across_repeats/mean/reward": 0.80,
                        "pass@1[avg-of-2]/accuracy": 75.0,
                    },
                    key_metrics={"pass@1[avg-of-2]/accuracy": 75.0},
                    groups=[_group(0, [1.0, 1.0]), _group(1, [0.0, 0.0])],
                )
            ],
        )
        candidate_rollouts = _write_run(
            tmp_path,
            "run_b",
            [
                _entry(
                    agent_metrics={"mean/reward": 0.25, "pass@1[avg-of-2]/accuracy": 25.0},
                    key_metrics={"pass@1[avg-of-2]/accuracy": 25.0},
                    groups=[_group(0, [0.0, 0.0]), _group(1, [0.0, 0.0])],
                )
            ],
        )
        config = ComparisonConfig.model_validate(
            {
                "baseline_rollouts_jsonl_fpath": str(baseline_rollouts),
                "candidate_rollouts_jsonl_fpaths": [str(candidate_rollouts)],
                **config_overrides,
            }
        )
        return config, build_comparison_result(config, "gym compare ...")

    def test_result_carries_both_sides_and_the_flip(self, tmp_path):
        _, result = self._result(tmp_path)
        assert result.schema_version == "1"
        assert result.baseline.label == "run_a" and result.candidates[0].label == "run_b"
        comparison = result.comparisons[0]
        key_row = next(row for row in comparison.metrics if row.is_key_metric)
        assert key_row.metric == "pass@1[avg-of-2]/accuracy"
        assert key_row.candidates[0].delta == pytest.approx(-50.0)
        assert comparison.flips[0].pass_to_fail_count == 1

    def test_output_dir_defaults_to_the_candidate_directory(self, tmp_path):
        config, _ = self._result(tmp_path)
        assert resolve_output_dir(config) == tmp_path / "run_b"

    def test_output_dir_flag_wins(self, tmp_path):
        config, _ = self._result(tmp_path, output_dirpath=str(tmp_path / "elsewhere"))
        assert resolve_output_dir(config) == tmp_path / "elsewhere"

    @pytest.mark.parametrize(
        "report_format, expected",
        [
            ("both", ["compare_report.md", "compare_report.json"]),
            ("md", ["compare_report.md"]),
            ("json", ["compare_report.json"]),
        ],
    )
    def test_report_format_selects_the_artifacts(self, tmp_path, report_format, expected):
        _, result = self._result(tmp_path)
        output_dir = tmp_path / f"report_{report_format}"
        written = write_reports(result, output_dir, report_format)
        assert [path.name for path in written] == expected
        assert all(path.exists() and path.stat().st_size > 0 for path in written)
        # Only the requested artifacts are written.
        assert sorted(path.name for path in output_dir.iterdir()) == sorted(expected)

    def test_an_unwritable_output_dir_is_rejected_cleanly(self, tmp_path):
        _, result = self._result(tmp_path)
        readonly = tmp_path / "readonly"
        readonly.mkdir()
        readonly.chmod(0o500)
        try:
            if os.access(readonly, os.W_OK):
                pytest.skip("cannot make a directory read-only as this user (running as root?)")
            # mkdir of a child fails...
            with pytest.raises(ConfigError, match="Cannot write to --output-dir"):
                write_reports(result, readonly / "nested", "both")
            # ...and so does writing into an existing but read-only directory.
            with pytest.raises(ConfigError, match="Cannot write the report into"):
                write_reports(result, readonly, "both")
        finally:
            readonly.chmod(0o755)

    def test_run_comparison_returns_the_result_and_the_paths_it_wrote(self, tmp_path):
        from nemo_gym.comparison.runner import run_comparison

        config, _ = self._result(tmp_path, output_dirpath=str(tmp_path / "out"))
        result, written = run_comparison(config, "gym eval compare ...")
        assert result.comparisons[0].baseline_agent == AGENT
        assert [path.name for path in written] == ["compare_report.md", "compare_report.json"]
        assert all(path.exists() for path in written)

    def test_invoked_command_records_the_resolved_invocation(self, monkeypatch):
        from nemo_gym.comparison.runner import invoked_command

        # `dispatch` rewrites argv to the resolved Hydra overrides before the entry point runs.
        monkeypatch.setattr("sys.argv", ["gym", "+baseline_rollouts_jsonl_fpath=a b.jsonl"])
        assert invoked_command() == "gym eval compare '+baseline_rollouts_jsonl_fpath=a b.jsonl'"

    def test_invoked_command_redacts_secret_shaped_overrides(self, monkeypatch):
        from nemo_gym.comparison.runner import invoked_command

        monkeypatch.setattr(
            "sys.argv",
            [
                "gym",
                "+baseline_rollouts_jsonl_fpath=a.jsonl",
                "++policy_api_key=sk-super-secret",
                "++some.nested.hf_token=hf_super_secret",
            ],
        )
        command = invoked_command()
        assert "sk-super-secret" not in command
        assert "hf_super_secret" not in command
        assert "+baseline_rollouts_jsonl_fpath=a.jsonl" in command
        assert "++policy_api_key=****" in command
        assert "++some.nested.hf_token=****" in command

    def test_key_metrics_table_is_rendered_per_agent(self, tmp_path):
        from nemo_gym.comparison.report import render_key_metrics_tables

        _, result = self._result(tmp_path)
        (table,) = render_key_metrics_tables(result)
        assert table.title == f"Key metrics — {AGENT}"
        assert [column.header for column in table.columns] == [
            "Metric",
            "Δ (cand − base)",
            "Baseline",
            "Baseline 95% CI",
            "Candidate",
            "Candidate 95% CI",
        ]
        # Only key metrics get a row, and `[avg-of-k]` survives Rich markup escaping.
        assert table.row_count == 1
        assert "pass@1\\[avg-of-2]/accuracy" in list(table.columns[0].cells)

    @pytest.mark.parametrize(
        "groups, expected",
        [
            ([_group(0, [0.25])], "Per-task changes: 1 of 1 common tasks moved"),
            ([_group(9, [1.0])], "Sample flips unavailable:"),
        ],
    )
    def test_summary_reports_every_flip_mode(self, tmp_path, groups, expected):
        baseline = _write_run(tmp_path / "modes", "run_a", [_entry(groups=[_group(0, [0.75])])])
        candidate = _write_run(tmp_path / "modes", "run_b", [_entry(groups=groups)])
        config = ComparisonConfig.model_validate(
            {
                "baseline_rollouts_jsonl_fpath": str(baseline),
                "candidate_rollouts_jsonl_fpaths": [str(candidate)],
            }
        )
        result = build_comparison_result(config, "gym eval compare ...")
        assert any(expected in line for line in summary_lines(result, []))

    def test_json_report_round_trips(self, tmp_path):
        _, result = self._result(tmp_path)
        (json_fpath,) = write_reports(result, tmp_path / "report", "json")
        payload = orjson.loads(json_fpath.read_bytes())
        assert payload["schema_version"] == "1"
        # Candidate-varying fields stay list-shaped so a second candidate needs no schema change.
        assert isinstance(payload["candidates"], list)
        assert isinstance(payload["comparisons"][0]["metrics"][0]["candidates"], list)
        assert payload == orjson.loads(result.model_dump_json())

    def test_output_dir_cannot_be_a_file(self, tmp_path):
        _, result = self._result(tmp_path)
        clash = tmp_path / "not_a_dir"
        clash.write_text("")
        with pytest.raises(ConfigError, match="is not a directory"):
            write_reports(result, clash, "md")

    def test_summary_mentions_flips_and_written_paths(self, tmp_path):
        _, result = self._result(tmp_path)
        written = write_reports(result, tmp_path / "report", "both")
        text = "\n".join(summary_lines(result, written))
        assert "Sample flips: 0 fail→pass, 1 pass→fail over 2 common tasks" in text
        assert "compare_report.md" in text

    def test_markdown_report_renders_the_expected_sections(self, tmp_path):
        _, result = self._result(tmp_path)
        markdown = render_markdown(result)
        assert "# gym eval compare" in markdown
        assert "### Key metrics" in markdown
        assert "| Metric | Δ (cand − base) | Baseline | Baseline 95% CI | Candidate | Candidate 95% CI |" in markdown
        # A metric with no recorded interval renders an em dash rather than a fabricated one.
        assert "| `pass@1[avg-of-2]/accuracy` | -50.00 (-66.7%) | 75.00 | — | 25.00 | — |" in markdown
        assert "| `mean/reward` | -0.5000 (-66.7%) | 0.7500 | [0.7000, 0.8000] | 0.2500 | — |" in markdown
        assert "### Sample flips" in markdown
        assert "1 pass→fail" in markdown


class TestReportEdgeCases:
    def _result(self, tmp_path, baseline_entry, candidate_entry, name="edge"):
        baseline_rollouts = _write_run(tmp_path / name, "run_a", [baseline_entry])
        candidate_rollouts = _write_run(tmp_path / name, "run_b", [candidate_entry])
        config = ComparisonConfig.model_validate(
            {
                "baseline_rollouts_jsonl_fpath": str(baseline_rollouts),
                "candidate_rollouts_jsonl_fpaths": [str(candidate_rollouts)],
            }
        )
        return build_comparison_result(config, "gym compare ...")

    def test_missing_values_and_zero_baseline_render_placeholders(self, tmp_path):
        baseline = _entry(
            agent_metrics={"mean/reward": 0.0, "pass@1/accuracy": 10.0},
            key_metrics={},
            groups=[_group(0, [0.0])],
        )
        candidate = _entry(
            agent_metrics={"mean/reward": 0.5},
            key_metrics={},
            groups=[_group(0, [0.0])],
        )
        markdown = render_markdown(self._result(tmp_path, baseline, candidate))
        # No key metrics were recorded, and the one-sided metric has no delta to show.
        assert "No key metrics were recorded for this agent." in markdown
        assert "| `pass@1/accuracy` | — | 10.00 | — | — | — |" in markdown
        # A zero baseline has no meaningful relative change.
        assert "| `mean/reward` | +0.5000 (n/a) |" in markdown
        assert "### Metrics present in only one run" in markdown

    def test_continuous_mode_section(self, tmp_path):
        baseline = _entry(groups=[_group(0, [0.20]), _group(1, [0.90])])
        candidate = _entry(groups=[_group(0, [0.55]), _group(1, [0.90])])
        markdown = render_markdown(self._result(tmp_path, baseline, candidate))
        assert "Rewards are not binary" in markdown
        assert "2 common tasks · 1 changed · 1 unchanged" in markdown

    def test_unavailable_flips_explain_themselves(self, tmp_path):
        baseline = _entry(groups=[_group(0, [1.0])])
        candidate = _entry(groups=[_group(9, [1.0])])
        markdown = render_markdown(self._result(tmp_path, baseline, candidate))
        assert "Not available: no overlapping task indices" in markdown

    def test_flip_table_is_capped_per_direction(self, tmp_path):
        """The cap applies to each direction, so improvements are never crowded out by regressions."""
        from nemo_gym.comparison.report import MAX_FLIPS_SHOWN

        over_cap = MAX_FLIPS_SHOWN + 3
        # Regressions first in task order, then the same number of improvements. A single global cap
        # would show only regressions; a per-direction cap shows MAX_FLIPS_SHOWN of each.
        baseline_groups = [_group(i, [1.0, 1.0]) for i in range(over_cap)]
        candidate_groups = [_group(i, [0.0, 0.0]) for i in range(over_cap)]
        baseline_groups += [_group(over_cap + i, [0.0, 0.0]) for i in range(over_cap)]
        candidate_groups += [_group(over_cap + i, [1.0, 1.0]) for i in range(over_cap)]

        markdown = render_markdown(
            self._result(tmp_path, _entry(groups=baseline_groups), _entry(groups=candidate_groups))
        )
        assert markdown.count("| pass→fail |") == MAX_FLIPS_SHOWN
        assert markdown.count("| fail→pass |") == MAX_FLIPS_SHOWN
        assert f"… and {2 * over_cap - 2 * MAX_FLIPS_SHOWN} more (full list in `compare_report.json`)." in markdown

    def test_binary_detection_falls_back_to_task_means(self, tmp_path):
        """Older runs may not record per-task min/max; the task mean still identifies 0/1 rewards."""
        baseline_groups = [_group(0, [1.0]), _group(1, [0.0])]
        candidate_groups = [_group(0, [0.0]), _group(1, [0.0])]
        for group in baseline_groups + candidate_groups:
            group.pop("min/reward")
            group.pop("max/reward")
        summary = build_flip_summary(
            _load(tmp_path / "fb", "base", [_entry(groups=baseline_groups)]),
            _load(tmp_path / "fb", "cand", [_entry(groups=candidate_groups)], role="candidate"),
        )
        assert summary.mode == "binary"
        assert summary.pass_to_fail_count == 1

    def test_binary_detection_survives_a_task_missing_only_one_of_min_or_max(self, tmp_path):
        """A task recording just one of min/reward, max/reward (not neither) must still be read as
        binary -- the missing companion must not make the whole comparison look continuous."""
        baseline_groups = [_group(0, [1.0, 1.0]), _group(1, [0.0, 0.0])]
        candidate_groups = [_group(0, [0.0, 0.0]), _group(1, [0.0, 0.0])]
        # Drop only `max/reward` from task 0's group on each side; `min/reward` is still present.
        baseline_groups[0].pop("max/reward")
        candidate_groups[0].pop("max/reward")
        summary = build_flip_summary(
            _load(tmp_path / "asym", "base", [_entry(groups=baseline_groups)]),
            _load(tmp_path / "asym", "cand", [_entry(groups=candidate_groups)], role="candidate"),
        )
        assert summary.mode == "binary"
        assert summary.pass_to_fail_count == 1

    def test_flip_rows_without_rollout_infos_render_placeholders(self, tmp_path):
        baseline = _entry(groups=[_group(0, [1.0], with_rollout_infos=False)])
        candidate = _entry(groups=[_group(0, [0.0], with_rollout_infos=False)])
        markdown = render_markdown(self._result(tmp_path, baseline, candidate))
        assert "| 0 | pass→fail | 1.0000 | 0.0000 | -1.0000 | — | — |" in markdown

    def test_multiple_agents_each_get_their_own_section(self, tmp_path):
        baseline = _write_run(
            tmp_path / "multi",
            "run_a",
            [_entry(agent="a", groups=[_group(0, [1.0])]), _entry(agent="b", groups=[_group(0, [1.0])])],
        )
        candidate = _write_run(
            tmp_path / "multi",
            "run_b",
            [_entry(agent="a", groups=[_group(0, [0.0])]), _entry(agent="b", groups=[_group(0, [0.0])])],
        )
        config = ComparisonConfig.model_validate(
            {
                "baseline_rollouts_jsonl_fpath": str(baseline),
                "candidate_rollouts_jsonl_fpaths": [str(candidate)],
            }
        )
        markdown = render_markdown(build_comparison_result(config, "gym compare ..."))
        assert "## Agent: `a`" in markdown
        assert "## Agent: `b`" in markdown

    def test_warnings_section_lists_skipped_agents(self, tmp_path):
        baseline = _write_run(
            tmp_path / "warn",
            "run_a",
            [_entry(agent="a", groups=[_group(0, [1.0])]), _entry(agent="extra", groups=[_group(0, [1.0])])],
        )
        candidate = _write_run(tmp_path / "warn", "run_b", [_entry(agent="a", groups=[_group(0, [0.0])])])
        config = ComparisonConfig.model_validate(
            {
                "baseline_rollouts_jsonl_fpath": str(baseline),
                "candidate_rollouts_jsonl_fpaths": [str(candidate)],
            }
        )
        result = build_comparison_result(config, "gym compare ...")
        assert result.skipped_agents == {"baseline": ["extra"]}
        markdown = render_markdown(result)
        assert "## Warnings" in markdown
        assert "extra" in markdown
