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

from pathlib import Path

import orjson

from nemo_gym.eval.comparison.report import render_markdown
from nemo_gym.eval.comparison.runner import build_comparison_result
from nemo_gym.eval.comparison.schema import ComparisonConfig, ComparisonResult
from nemo_gym.metrics.reward_profile import compute_aggregate_metrics, compute_pass_majority_metrics
from nemo_gym.utils.path_utils import aggregate_metrics_path_for


def _entry(agent: str, *, masked: bool = False, with_perf: bool = True) -> dict:
    rows = [
        {
            "_ng_task_index": task,
            "_ng_rollout_index": repeat,
            "reward": (repeat + task) % 2,
            "mask_sample": masked and repeat == 1,
            **({"ng_perf": {"completion_tokens": 10 * (repeat + 1), "num_turns": 1}} if with_perf else {}),
        }
        for repeat in range(3)
        for task in range(2)
    ]
    return {"agent_ref": {"name": agent}, **compute_aggregate_metrics(rows).model_dump(mode="json")}


def _write_source(directory: Path, entries: list[dict]) -> str:
    directory.mkdir()
    rollouts = directory / "rollouts.jsonl"
    rollouts.write_text("aggregate artifacts are sufficient", encoding="utf-8")
    aggregate_metrics_path_for(rollouts).write_bytes(orjson.dumps(entries))
    return str(rollouts)


def _compare(tmp_path: Path, entries: list[dict], *, agent: str | None = None) -> ComparisonResult:
    sides = [
        [_write_source(tmp_path / f"{side}-{index}", entries) for index in range(2)]
        for side in ("baseline", "candidate")
    ]
    return build_comparison_result(
        ComparisonConfig(
            baseline_rollouts_jsonl_fpaths=sides[0],
            pooled_candidate_rollouts_jsonl_fpaths=sides[1],
            agent_name=agent,
        ),
        "gym eval compare",
    )


def test_unused_malformed_agent_is_not_loaded_for_pooled_provenance(tmp_path: Path) -> None:
    unused = {
        "agent_ref": {"name": "unused"},
        "repeat_level_metrics": [],
        "group_level_metrics": [1, 2],
    }
    result = _compare(tmp_path, [_entry("selected"), unused], agent="selected")

    assert result.comparisons[0].baseline_agent == "selected"
    reward = next(row for row in result.comparisons[0].metrics if row.metric == "mean/reward")
    assert reward.candidates[0].delta == 0
    for run in (result.baseline, *result.candidates):
        for source in run.sources:
            assert source.selected_agents == ["selected"]
            assert source.repeat_counts_by_agent == {"selected": 3}
            assert source.scored_repeat_counts_by_agent == {"selected": 3}


def test_masked_quality_counts_and_multiagent_source_tables_are_unambiguous(tmp_path: Path) -> None:
    result = _compare(tmp_path, [_entry("first", masked=True), _entry("second", masked=True)])
    assert [comparison.baseline_agent for comparison in result.comparisons] == ["first", "second"]
    for comparison in result.comparisons:
        assert comparison.baseline_repeat_count == 4
        assert comparison.candidate_repeat_counts == [4]
        for run in (result.baseline, *result.candidates):
            assert sum(source.scored_repeat_counts_by_agent[comparison.baseline_agent] for source in run.sources) == 4
            assert all(source.repeat_counts_by_agent[comparison.baseline_agent] == 3 for source in run.sources)

    markdown = render_markdown(result)
    assert markdown.count("### Baseline pooled sources") == 1
    assert markdown.count("### Candidate pooled sources") == 1
    assert "recorded repeat rows / scored repeats (tasks)" in markdown
    assert "`first`: 3 / 2 (2)" in markdown
    assert "`second`: 3 / 2 (2)" in markdown
    serialized = result.model_dump(mode="json")
    assert serialized["baseline"]["sources"][0]["scored_repeat_counts_by_agent"] == {"first": 2, "second": 2}


def test_pooled_notes_identify_the_side_of_each_observation(tmp_path: Path) -> None:
    result = _compare(tmp_path, [_entry("agent")])
    notes = result.comparisons[0].notes

    assert sum(note.startswith("Baseline: Pooled 2 independent sources") for note in notes) == 1
    assert sum(note.startswith("Candidate: Pooled 2 independent sources") for note in notes) == 1
    assert all(note.startswith(("Baseline:", "Candidate:")) for note in notes)
    assert len(notes) == len(set(notes))


def test_whole_run_key_metrics_are_reported_as_unpoolable(tmp_path: Path) -> None:
    entry = _entry("agent", with_perf=False)
    metric = "pass@1[avg-of-3]/accuracy"
    benchmark_metrics, _, _, _ = compute_pass_majority_metrics(
        [group["rollout_infos"] for group in entry["group_level_metrics"]]
    )
    entry["agent_metrics"].update(benchmark_metrics)
    entry["key_metrics"] = {metric: benchmark_metrics[metric]}
    result = _compare(tmp_path, [entry])

    assert not any(row.is_key_metric for row in result.comparisons[0].metrics)
    assert any(metric in note and "omitted" in note for note in result.comparisons[0].notes)
    markdown = render_markdown(result)
    assert "No key metrics could be pooled for this agent; see Notes." in markdown
    assert "No key metrics were recorded for this agent." not in markdown
