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

from math import sqrt
from pathlib import Path
from statistics import fmean, variance

import orjson
import pytest
from scipy.stats import t, ttest_ind

from nemo_gym.config.config_types import ConfigError
from nemo_gym.eval.comparison.loading import LoadedRun
from nemo_gym.eval.comparison.pooling import pool_loaded_runs
from nemo_gym.eval.comparison.report import render_markdown, summary_lines, write_reports
from nemo_gym.eval.comparison.runner import build_comparison_result, resolve_output_dir
from nemo_gym.eval.comparison.schema import ComparisonConfig
from nemo_gym.metrics.reward_profile import compute_aggregate_metrics
from nemo_gym.utils.path_utils import aggregate_metrics_path_for


METRIC = "mean/reward"


def _entry(values: list[float], *, agent: str = "agent") -> dict:
    return {
        "agent_ref": {"name": agent},
        # Deliberately inconsistent aggregate summaries prove the pooled path uses observations.
        "agent_metrics": {METRIC: 99, "pass@1[avg-of-8]/accuracy": 99},
        "key_metrics": {METRIC: 99, "pass@1[avg-of-8]/accuracy": 99},
        "repeat_level_metrics": [
            {"_ng_rollout_index": index, "sample_count": 1, "missing_count": 0, METRIC: value}
            for index, value in enumerate(values)
        ],
        "group_level_metrics": [
            {
                "_ng_task_index": 0,
                "expected_num_rollouts": len(values),
                "mean/reward": fmean(values),
                "min/reward": min(values),
                "max/reward": max(values),
                "rollout_infos": [
                    {"_ng_task_index": 0, "_ng_rollout_index": index, "reward": value}
                    for index, value in enumerate(values)
                ],
            }
        ],
    }


def _write_run(tmp_path: Path, name: str, entries: list[dict]) -> str:
    path = tmp_path / name / "rollouts.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    # The aggregation artifacts are sufficient; compare must never read this file.
    path.write_text("not JSONL", encoding="utf-8")
    aggregate_metrics_path_for(path).write_bytes(orjson.dumps(entries))
    return str(path)


def _config(tmp_path: Path, *, baseline_entries=None, candidate_entries=None, **overrides) -> ComparisonConfig:
    baseline = baseline_entries or [[_entry([0.1, 0.3])], [_entry([0.6, 0.8, 0.9])]]
    candidate = candidate_entries or [[_entry([0.2, 0.4, 0.6])], [_entry([0.7, 0.8])], [_entry([0.9, 1])]]
    return ComparisonConfig(
        baseline_rollouts_jsonl_fpaths=[
            _write_run(tmp_path, f"base{i}", entries) for i, entries in enumerate(baseline)
        ],
        pooled_candidate_rollouts_jsonl_fpaths=[
            _write_run(tmp_path, f"cand{i}", entries) for i, entries in enumerate(candidate)
        ],
        **overrides,
    )


def _reward_row(result):
    return next(row for row in result.comparisons[0].metrics if row.metric == METRIC)


def test_pooling_recomputes_unequal_samples_and_welch_interval(tmp_path):
    config = _config(tmp_path, confidence_level=0.90)
    result = build_comparison_result(config, "gym eval compare")
    row = _reward_row(result)
    baseline, candidate = [0.1, 0.3, 0.6, 0.8, 0.9], [0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1]
    baseline_variance = variance(baseline) / len(baseline)
    candidate_variance = variance(candidate) / len(candidate)
    delta = fmean(candidate) - fmean(baseline)
    degrees_of_freedom = (baseline_variance + candidate_variance) ** 2 / (
        baseline_variance**2 / (len(baseline) - 1) + candidate_variance**2 / (len(candidate) - 1)
    )
    margin = t.ppf(0.95, degrees_of_freedom) * sqrt(baseline_variance + candidate_variance)
    assert row.baseline.value == pytest.approx(fmean(baseline))
    assert row.baseline.se_across_repeats == pytest.approx(sqrt(baseline_variance))
    baseline_margin = t.ppf(0.975, len(baseline) - 1) * sqrt(baseline_variance)
    assert row.baseline.ci_low == pytest.approx(fmean(baseline) - baseline_margin)
    assert row.baseline.ci_high == pytest.approx(fmean(baseline) + baseline_margin)
    assert row.candidates[0].value == pytest.approx(fmean(candidate))
    assert row.candidates[0].delta == pytest.approx(delta)
    assert row.candidates[0].delta_ci_low == pytest.approx(delta - margin)
    assert row.candidates[0].delta_ci_high == pytest.approx(delta + margin)
    comparison = result.comparisons[0]
    assert comparison.baseline_repeat_count == 5
    assert comparison.candidate_repeat_counts == [7]
    assert len(result.candidates) == 1
    assert [row.metric for row in comparison.metrics] == [METRIC]
    assert any("pass@1[avg-of-8]/accuracy" in note and "omitted" in note for note in comparison.notes)
    assert not any("Repeat counts differ" in note for note in comparison.notes)


def test_sources_paths_counts_and_pooling_semantics_survive_reports(tmp_path):
    config = _config(tmp_path)
    result = build_comparison_result(config, "gym eval compare")
    assert [source.repeat_counts_by_agent for source in result.baseline.sources] == [{"agent": 2}, {"agent": 3}]
    assert [source.repeat_counts_by_agent for source in result.candidates[0].sources] == [
        {"agent": 3},
        {"agent": 2},
        {"agent": 2},
    ]
    markdown = render_markdown(result)
    assert "### Baseline pooled sources" in markdown
    assert "### Candidate pooled sources" in markdown
    assert "| Repeats | 5 | 7 |" in markdown
    assert "recomputed from all concatenated repeat observations" in markdown
    recap = "\n".join(summary_lines(result, []))
    for run_path in [*config.baseline_rollouts_jsonl_fpaths, *config.pooled_candidate_rollouts_jsonl_fpaths]:
        assert f"`{run_path}`" in markdown
        assert str(aggregate_metrics_path_for(Path(run_path))) in recap
    written = write_reports(result, tmp_path / "reports", "both")
    payload = orjson.loads(next(path for path in written if path.suffix == ".json").read_bytes())
    assert len(payload["baseline"]["sources"]) == 2
    assert len(payload["candidates"]) == 1
    assert len(payload["candidates"][0]["sources"]) == 3
    assert resolve_output_dir(config) == Path(config.pooled_candidate_rollouts_jsonl_fpaths[-1]).parent


@pytest.mark.parametrize("pooled_side", ["baseline", "candidate"])
def test_either_side_can_be_pooled_with_an_existing_single_run(tmp_path, pooled_side):
    pooled = [_write_run(tmp_path, f"pool{i}", [_entry(values)]) for i, values in enumerate(([0, 0.4], [0.7, 1]))]
    single_entry = _entry([0.3, 0.6])
    single_entry["agent_metrics"].update({METRIC: 0.45, "mean_across_repeats/mean/reward": 0.45})
    single = _write_run(tmp_path, "single", [single_entry])
    if pooled_side == "baseline":
        config = ComparisonConfig(baseline_rollouts_jsonl_fpaths=pooled, candidate_rollouts_jsonl_fpaths=[single])
    else:
        config = ComparisonConfig(baseline_rollouts_jsonl_fpath=single, pooled_candidate_rollouts_jsonl_fpaths=pooled)
    result = build_comparison_result(config, "gym eval compare")
    row = _reward_row(result)
    expected_delta = 0.45 - 0.525 if pooled_side == "baseline" else 0.525 - 0.45
    assert row.candidates[0].delta == pytest.approx(expected_delta)
    assert row.candidates[0].delta_ci_low is not None
    assert bool(result.baseline.sources) == (pooled_side == "baseline")
    assert bool(result.candidates[0].sources) == (pooled_side == "candidate")


def test_default_agents_must_be_common_to_every_constituent(tmp_path):
    result = build_comparison_result(
        _config(
            tmp_path,
            baseline_entries=[[_entry([0, 1]), _entry([0, 1], agent="first_only")], [_entry([0, 1])]],
            candidate_entries=[[_entry([0, 1]), _entry([0, 1], agent="candidate_only")], [_entry([0, 1])]],
        ),
        "gym eval compare",
    )
    assert [comparison.baseline_agent for comparison in result.comparisons] == ["agent"]
    assert result.skipped_agents == {"baseline": ["first_only"], "candidate[0]": ["candidate_only"]}


def test_per_source_agent_choices_and_aggregate_overrides_are_honored(tmp_path):
    config = _config(
        tmp_path,
        baseline_entries=[[_entry([0.1, 0.2], agent="base_a")], [_entry([0.3, 0.4], agent="base_b")]],
        candidate_entries=[[_entry([0.5, 0.6], agent="cand_a")], [_entry([0.7, 0.8], agent="cand_b")]],
        baseline_agent_names=["base_a", "base_b"],
        candidate_agent_names=["cand_a", "cand_b"],
    )
    baseline_overrides, candidate_overrides = [], []
    for paths, overrides in (
        (config.baseline_rollouts_jsonl_fpaths, baseline_overrides),
        (config.pooled_candidate_rollouts_jsonl_fpaths, candidate_overrides),
    ):
        for path in paths:
            original = aggregate_metrics_path_for(Path(path))
            target = original.with_name("explicit.json")
            original.rename(target)
            overrides.append(str(target))
    config.baseline_aggregate_metrics_fpaths = baseline_overrides
    config.candidate_aggregate_metrics_fpaths = candidate_overrides
    result = build_comparison_result(config, "gym eval compare")
    row = _reward_row(result)
    assert row.baseline.value == pytest.approx(0.25)
    assert row.candidates[0].value == pytest.approx(0.65)
    assert result.comparisons[0].baseline_agent == "base_a, base_b"
    assert result.comparisons[0].candidate_agents == ["cand_a, cand_b"]
    assert [source.selected_agents for source in result.baseline.sources] == [["base_a"], ["base_b"]]
    assert [source.selected_agents for source in result.candidates[0].sources] == [["cand_a"], ["cand_b"]]
    assert [str(source.aggregate_metrics_fpath) for source in result.baseline.sources] == baseline_overrides


@pytest.mark.parametrize("bad_value", [None, "unknown", True])
def test_incomplete_metric_never_silently_drops_a_source(tmp_path, bad_value):
    broken = _entry([0.2, 0.3])
    broken["repeat_level_metrics"][0][METRIC] = bad_value
    config = _config(tmp_path, baseline_entries=[[_entry([0, 1])], [broken]])
    with pytest.raises(ConfigError, match="Cannot pool metric 'mean/reward'.*base1.*repeat 0"):
        build_comparison_result(config, "gym eval compare")


@pytest.mark.parametrize("bad_value", [float("inf"), float("-inf"), float("nan")])
def test_programmatic_pooling_rejects_nonfinite_observations(bad_value: float) -> None:
    entry = _entry([0.2, 0.3])
    entry["repeat_level_metrics"][0][METRIC] = bad_value
    run = LoadedRun(
        agent_name="agent",
        agent_metrics=entry["agent_metrics"],
        key_metrics=entry["key_metrics"],
        group_level_metrics=entry["group_level_metrics"],
        repeat_level_metrics=entry["repeat_level_metrics"],
    )
    with pytest.raises(ConfigError, match="Cannot pool metric 'mean/reward'.*nonfinite-source.*repeat 0"):
        pool_loaded_runs([run], source_labels=["nonfinite-source"])


def test_metric_absent_from_entire_constituent_is_an_error(tmp_path):
    broken = _entry([0.2, 0.3])
    for row in broken["repeat_level_metrics"]:
        del row[METRIC]
    config = _config(tmp_path, baseline_entries=[[_entry([0, 1])], [broken]])
    with pytest.raises(ConfigError, match="Every constituent must provide every pooled metric"):
        build_comparison_result(config, "gym eval compare")


@pytest.mark.parametrize(
    "rows,message",
    [
        ([], "no repeat-level observations"),
        ([None], "repeat row 0 is not an object"),
        ([{METRIC: 1}], "nonnegative integer _ng_rollout_index"),
        ([{"_ng_rollout_index": True, METRIC: 1}], "nonnegative integer _ng_rollout_index"),
        ([{"_ng_rollout_index": -1, METRIC: 1}], "nonnegative integer _ng_rollout_index"),
        ([{"_ng_rollout_index": 0, METRIC: 0}] * 2, "duplicate local repeat index"),
        ([{"_ng_rollout_index": 0, METRIC: 0, "sample_count": -1}], "invalid sample_count"),
    ],
)
def test_malformed_repeat_rows_are_rejected_with_source_context(tmp_path, rows, message):
    broken = _entry([0.2, 0.3])
    broken["repeat_level_metrics"] = rows
    config = _config(tmp_path, baseline_entries=[[broken], [_entry([0, 1])]])
    with pytest.raises(ConfigError, match=message) as error:
        build_comparison_result(config, "gym eval compare")
    assert "base0" in str(error.value)


def test_flips_recompute_task_rewards_using_all_observations(tmp_path):
    result = build_comparison_result(
        _config(
            tmp_path,
            baseline_entries=[[_entry([1, 1, 1])], [_entry([0, 0])]],
            candidate_entries=[[_entry([0, 0, 0])], [_entry([0, 0])]],
        ),
        "gym eval compare",
    )
    flip = result.comparisons[0].flips[0]
    assert flip.mode == "binary"
    assert flip.pass_to_fail_count == 1
    assert flip.flips[0].baseline_score == pytest.approx(3 / 5)
    assert flip.flips[0].baseline_rewards == [1, 1, 1, 0, 0]
    assert flip.flips[0].candidate_rewards == [0, 0, 0, 0, 0]


def test_flips_are_unavailable_when_source_only_carries_task_means(tmp_path):
    broken = _entry([0, 1])
    del broken["group_level_metrics"][0]["rollout_infos"]
    result = build_comparison_result(_config(tmp_path, baseline_entries=[[broken]]), "gym eval compare")
    flip = result.comparisons[0].flips[0]
    assert flip.mode == "unavailable"
    assert "stored rollout_infos" in flip.reason
    assert _reward_row(result).baseline.value == 0.5


@pytest.mark.parametrize("with_rollout_infos", [False, True])
def test_different_task_sets_warn_even_with_equal_sample_counts(tmp_path, with_rollout_infos):
    different_task = _entry([0, 1])
    different_task["group_level_metrics"][0]["_ng_task_index"] = 10
    for info in different_task["group_level_metrics"][0]["rollout_infos"]:
        info["_ng_task_index"] = 10
    if not with_rollout_infos:
        del different_task["group_level_metrics"][0]["rollout_infos"]
    result = build_comparison_result(
        _config(tmp_path, baseline_entries=[[_entry([0, 1])], [different_task]]), "gym eval compare"
    )
    comparison = result.comparisons[0]
    assert any("task-difficulty differences" in note for note in comparison.notes)
    assert comparison.baseline_task_count == 2
    assert comparison.flips[0].mode == "unavailable"
    assert ("different task indices" if with_rollout_infos else "stored rollout_infos") in comparison.flips[0].reason


@pytest.mark.parametrize("reuse_kind", ["within_side", "aggregate_symlink", "rollout_symlink"])
def test_reused_artifacts_are_rejected_including_filesystem_aliases(tmp_path, reuse_kind):
    config = _config(tmp_path)
    baseline_path = Path(config.baseline_rollouts_jsonl_fpaths[0])
    candidate_path = Path(config.pooled_candidate_rollouts_jsonl_fpaths[0])
    if reuse_kind == "within_side":
        config.baseline_rollouts_jsonl_fpaths[1] = str(baseline_path)
    elif reuse_kind == "aggregate_symlink":
        aggregate_metrics_path_for(candidate_path).unlink()
        aggregate_metrics_path_for(candidate_path).symlink_to(aggregate_metrics_path_for(baseline_path))
    else:
        candidate_path.unlink()
        candidate_path.symlink_to(baseline_path)
        # Retain distinct aggregate overrides to exercise rollout identity independently.
        config.candidate_aggregate_metrics_fpaths = [
            str(aggregate_metrics_path_for(Path(path))) for path in config.pooled_candidate_rollouts_jsonl_fpaths
        ]
    with pytest.raises(ConfigError, match="reused.*independent run"):
        build_comparison_result(config, "gym eval compare")


@pytest.mark.parametrize(
    "payload,message",
    [
        ({"candidate_rollouts_jsonl_fpaths": ["c"]}, "exactly one of --baseline"),
        ({"baseline_rollouts_jsonl_fpath": "b"}, "exactly one of --candidates"),
        (
            {
                "baseline_rollouts_jsonl_fpath": "b",
                "baseline_rollouts_jsonl_fpaths": ["b2"],
                "candidate_rollouts_jsonl_fpaths": ["c"],
            },
            "exactly one of --baseline",
        ),
        (
            {
                "baseline_rollouts_jsonl_fpath": "b",
                "candidate_rollouts_jsonl_fpaths": ["c"],
                "pooled_candidate_rollouts_jsonl_fpaths": ["c2"],
            },
            "exactly one of --candidates",
        ),
        (
            {
                "baseline_rollouts_jsonl_fpaths": ["b"],
                "candidate_rollouts_jsonl_fpaths": ["c"],
                "baseline_agent_names": ["a", "b"],
            },
            "one entry per --baseline-runs source",
        ),
        (
            {
                "baseline_rollouts_jsonl_fpath": "b",
                "pooled_candidate_rollouts_jsonl_fpaths": ["c", "d"],
                "candidate_agent_names": ["a"],
            },
            "candidate_agent_names has 1 entries but 2",
        ),
        (
            {
                "baseline_rollouts_jsonl_fpaths": ["b"],
                "candidate_rollouts_jsonl_fpaths": ["c"],
                "baseline_aggregate_metrics_fpath": "override.json",
            },
            "Use baseline_aggregate_metrics_fpaths",
        ),
    ],
)
def test_grouped_input_contracts_are_validated(payload, message):
    with pytest.raises(ValueError, match=message):
        ComparisonConfig(**payload)


def test_existing_candidates_argument_still_means_separate_candidates():
    with pytest.raises(ValueError, match="comparing more than 1 candidate is not supported yet"):
        ComparisonConfig(baseline_rollouts_jsonl_fpath="base", candidate_rollouts_jsonl_fpaths=["c1", "c2"])


def test_existing_same_file_different_agents_remains_valid(tmp_path):
    shared = _write_run(tmp_path, "shared", [_entry([0, 0.2], agent="baseline"), _entry([0.5, 1], agent="candidate")])
    config = ComparisonConfig(
        baseline_rollouts_jsonl_fpath=shared,
        candidate_rollouts_jsonl_fpaths=[shared],
        baseline_agent_name="baseline",
        candidate_agent_names=["candidate"],
    )
    result = build_comparison_result(config, "gym eval compare")
    assert result.comparisons[0].baseline_agent == "baseline"
    assert result.comparisons[0].candidate_agents == ["candidate"]
    assert not result.baseline.sources


def test_pooled_sides_can_select_different_agents_from_shared_files(tmp_path):
    paths = [
        _write_run(
            tmp_path,
            f"shared-{index}",
            [_entry([0.1, 0.3], agent="baseline"), _entry([0.6, 0.8], agent="candidate")],
        )
        for index in range(2)
    ]
    result = build_comparison_result(
        ComparisonConfig(
            baseline_rollouts_jsonl_fpaths=paths,
            pooled_candidate_rollouts_jsonl_fpaths=paths,
            baseline_agent_names=["baseline", "baseline"],
            candidate_agent_names=["candidate", "candidate"],
        ),
        "gym eval compare",
    )
    assert result.comparisons[0].baseline_repeat_count == 4
    assert result.comparisons[0].candidate_repeat_counts == [4]
    assert _reward_row(result).candidates[0].delta == pytest.approx(0.5)
    assert [source.selected_agents for source in result.baseline.sources] == [["baseline"], ["baseline"]]
    assert [source.selected_agents for source in result.candidates[0].sources] == [["candidate"], ["candidate"]]


def _producer_entry(
    *, offset: float, with_perf: bool, scored_repeats: tuple[int, ...], perf_repeats: tuple[int, ...] = (0, 1, 2)
) -> dict:
    rows = []
    for repeat in range(3):
        for task in range(2):
            row = {
                "_ng_task_index": task,
                "_ng_rollout_index": repeat,
                "reward": 0.2 + 0.1 * repeat + 0.01 * task + offset,
            }
            if repeat not in scored_repeats:
                row["mask_sample"] = True
            if with_perf and repeat in perf_repeats:
                row["ng_perf"] = {"completion_tokens": 10 * (repeat + 1), "num_turns": 1}
            rows.append(row)
    return {"agent_ref": {"name": "agent"}, **compute_aggregate_metrics(rows).model_dump(mode="json")}


@pytest.mark.parametrize("missing_repeats", [(0, 1, 2), (1,)])
def test_missing_token_accounting_preserves_quality_samples(tmp_path: Path, missing_repeats: tuple[int, ...]) -> None:
    kept_repeats = tuple(index for index in range(3) if index not in missing_repeats)
    baseline = [
        [_producer_entry(offset=0, with_perf=True, scored_repeats=(0, 1, 2))],
        [_producer_entry(offset=0.04, with_perf=True, scored_repeats=(0, 1, 2), perf_repeats=kept_repeats)],
    ]
    config = _config(
        tmp_path,
        baseline_entries=baseline,
        candidate_entries=[[_producer_entry(offset=0.1, with_perf=True, scored_repeats=(0, 1, 2))]],
    )
    before = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    result = build_comparison_result(config, "gym eval compare")
    comparison = result.comparisons[0]
    baseline_sample = [0.205 + 0.1 * repeat + offset for offset in (0, 0.04) for repeat in range(3)]
    candidate_sample = [0.305 + 0.1 * repeat for repeat in range(3)]
    interval = ttest_ind(candidate_sample, baseline_sample, equal_var=False).confidence_interval(0.95)
    reward = _reward_row(result)
    assert comparison.baseline_repeat_count == 6
    assert comparison.candidate_repeat_counts == [3]
    assert reward.baseline.value == pytest.approx(fmean(baseline_sample))
    assert reward.candidates[0].delta_ci_low == pytest.approx(interval.low)
    assert reward.candidates[0].delta_ci_high == pytest.approx(interval.high)
    tokens = next(row for row in comparison.metrics if row.metric == "mean_completion_tokens")
    token_sample = [10, 20, 30] + [10 * (repeat + 1) for repeat in kept_repeats]
    assert tokens.baseline.value == pytest.approx(fmean(token_sample))
    assert tokens.baseline.se_across_repeats == pytest.approx(sqrt(variance(token_sample) / len(token_sample)))
    assert any(
        str(aggregate_metrics_path_for(Path(config.baseline_rollouts_jsonl_fpaths[1]))) in note
        and "mean_completion_tokens" in note
        and str(list(missing_repeats)) in note
        for note in comparison.notes
    )
    assert all(path.read_bytes() == contents for path, contents in before.items())


@pytest.mark.parametrize("bad_value", [None, "unknown", True])
def test_present_malformed_token_accounting_is_rejected(tmp_path: Path, bad_value: object) -> None:
    entry = _producer_entry(offset=0, with_perf=True, scored_repeats=(0, 1, 2))
    entry["repeat_level_metrics"][0]["mean_completion_tokens"] = bad_value
    config = _config(tmp_path, baseline_entries=[[entry]])
    with pytest.raises(ConfigError, match="Cannot pool metric 'mean_completion_tokens'.*base0.*repeat 0"):
        build_comparison_result(config, "gym eval compare")


@pytest.mark.parametrize("with_perf", [False, True])
def test_single_repeat_producer_error_explains_collection_requirement(tmp_path: Path, with_perf: bool) -> None:
    rows = [{"_ng_task_index": 0, "_ng_rollout_index": 0, "reward": 0.5}]
    if with_perf:
        rows[0]["ng_perf"] = {"completion_tokens": 10, "num_turns": 1}
    entry = {"agent_ref": {"name": "agent"}, **compute_aggregate_metrics(rows).model_dump(mode="json")}
    config = _config(tmp_path, baseline_entries=[[entry]])
    with pytest.raises(ConfigError, match="(?i)single-repeat") as error:
        build_comparison_result(config, "gym eval compare")
    assert "base0" in str(error.value)
    assert "two scored repeats" in str(error.value)


@pytest.mark.parametrize("with_perf", [False, True])
@pytest.mark.parametrize("scored_repeats", [(0, 1, 2), (0, 2)])
def test_producer_token_accounting_does_not_change_pooled_quality_samples(
    tmp_path: Path, with_perf: bool, scored_repeats: tuple[int, ...]
) -> None:
    baseline_offsets = [0.0, 0.04]
    candidate_offsets = [0.1, 0.17]
    config = _config(
        tmp_path,
        baseline_entries=[
            [_producer_entry(offset=offset, with_perf=with_perf, scored_repeats=scored_repeats)]
            for offset in baseline_offsets
        ],
        candidate_entries=[
            [_producer_entry(offset=offset, with_perf=with_perf, scored_repeats=scored_repeats)]
            for offset in candidate_offsets
        ],
    )
    source_bytes = {path: path.read_bytes() for path in tmp_path.rglob("*.json")}
    result = build_comparison_result(config, "gym eval compare")
    baseline = [0.205 + 0.1 * repeat + offset for offset in baseline_offsets for repeat in scored_repeats]
    candidate = [0.205 + 0.1 * repeat + offset for offset in candidate_offsets for repeat in scored_repeats]
    interval = ttest_ind(candidate, baseline, equal_var=False).confidence_interval(0.95)
    reward = _reward_row(result)
    comparison = result.comparisons[0]
    assert comparison.baseline_repeat_count == comparison.candidate_repeat_counts[0] == 2 * len(scored_repeats)
    assert reward.baseline.value == pytest.approx(fmean(baseline))
    assert reward.candidates[0].value == pytest.approx(fmean(candidate))
    assert reward.candidates[0].delta_ci_low == pytest.approx(interval.low)
    assert reward.candidates[0].delta_ci_high == pytest.approx(interval.high)
    if len(scored_repeats) < 3:
        assert any("Unscored repeats" in note and "excluded from quality" in note for note in comparison.notes)
    if with_perf:
        tokens = next(row for row in comparison.metrics if row.metric == "mean_completion_tokens")
        assert tokens.baseline.value == tokens.candidates[0].value == 20
        token_values = [10, 20, 30] * 2
        token_se = sqrt(variance(token_values) / len(token_values))
        assert tokens.baseline.se_across_repeats == pytest.approx(token_se)
        assert tokens.candidates[0].se_across_repeats == pytest.approx(token_se)
        if len(scored_repeats) < 3:
            for path in (*config.baseline_rollouts_jsonl_fpaths, *config.pooled_candidate_rollouts_jsonl_fpaths):
                assert any(
                    str(aggregate_metrics_path_for(Path(path))) in note
                    and "agent 'agent'" in note
                    and "local repeats [1]" in note
                    and "excluded from quality" in note
                    for note in comparison.notes
                )
    assert all(path.read_bytes() == contents for path, contents in source_bytes.items())


@pytest.mark.parametrize("with_perf", [False, True])
def test_producer_without_quality_repeat_estimates_never_hides_reward(tmp_path: Path, with_perf: bool) -> None:
    config = _config(
        tmp_path,
        baseline_entries=[
            [_producer_entry(offset=offset, with_perf=with_perf, scored_repeats=(0,))] for offset in (0.0, 0.1)
        ],
    )
    with pytest.raises(ConfigError, match="no (repeat-level observations|stored quality repeat-level estimates)"):
        build_comparison_result(config, "gym eval compare")


@pytest.mark.parametrize("collision", ["metrics_in_output_dir", "output_symlink"])
def test_output_collision_with_later_source_is_rejected_before_any_write(tmp_path, collision):
    config = _config(tmp_path)
    output_dir = tmp_path / "reports"
    output_dir.mkdir()
    later_source = aggregate_metrics_path_for(Path(config.baseline_rollouts_jsonl_fpaths[1]))
    original = later_source.read_bytes()
    destination = output_dir / "compare_report.json"
    if collision == "metrics_in_output_dir":
        later_source.rename(destination)
        config.baseline_aggregate_metrics_fpaths = [
            str(aggregate_metrics_path_for(Path(config.baseline_rollouts_jsonl_fpaths[0]))),
            str(destination),
        ]
        protected = destination
    else:
        destination.symlink_to(later_source)
        protected = later_source
    result = build_comparison_result(config, "gym eval compare")
    with pytest.raises(ConfigError, match="would overwrite a comparison input"):
        write_reports(result, output_dir, "both")
    assert protected.read_bytes() == original
    assert not (output_dir / "compare_report.md").exists()
