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
"""Exercise pooled-run commands against artifacts from the real metric producer."""

import json
from pathlib import Path
from statistics import mean

import pytest
import yaml
from hydra.core.override_parser.overrides_parser import OverridesParser
from scipy.stats import ttest_ind

from nemo_gym.cli.main import build_parser, main
from nemo_gym.metrics.reward_profile import compute_aggregate_metrics
from nemo_gym.utils.path_utils import aggregate_metrics_path_for, materialized_path_for


def test_pooling_flags_preserve_paths_and_selector_order() -> None:
    options = {
        "baseline-runs": ("baseline_rollouts_jsonl_fpaths", ["runs/[v2] baseline.jsonl", "b2.jsonl"]),
        "candidate-runs": ("pooled_candidate_rollouts_jsonl_fpaths", ["c1.jsonl", "c2.jsonl", "c3.jsonl"]),
        "baseline-runs-agg-metrics": ("baseline_aggregate_metrics_fpaths", ["b1 metrics.json", "b2 metrics.json"]),
        "candidates-agg-metrics": ("candidate_aggregate_metrics_fpaths", ["c1.json", "c2.json", "c3.json"]),
        "baseline-agents": ("baseline_agent_names", ["b1", "b2"]),
        "candidate-agents": ("candidate_agent_names", ["c1", "c2", "c3"]),
    }
    argv = ["eval", "compare"]
    for flag, (_, values) in options.items():
        argv.extend([f"--{flag}", ",".join(values)])
    args, unknown = build_parser().parse_known_args(argv)
    assert unknown == []
    overrides = [token for flag in args._command.flags for token in flag.translate_to_hydra(args)]
    parsed = OverridesParser.create().parse_overrides(overrides)
    assert {override.key_or_group: override.value() for override in parsed} == dict(options.values())


def _write_source(directory: Path, values: list[float]) -> Path:
    directory.mkdir()
    rows = [
        {
            "agent_ref": {"name": "agent"},
            "_ng_task_index": task,
            "_ng_rollout_index": repeat,
            "reward": value + (task - 0.5) * 0.02,
        }
        for repeat, value in enumerate(values)
        for task in range(2)
    ]
    rollouts = directory / "rollouts.jsonl"
    rollouts.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    aggregate = {"agent_ref": {"name": "agent"}, **compute_aggregate_metrics(rows).model_dump(mode="json")}
    aggregate_metrics_path_for(rollouts).write_text(json.dumps([aggregate]), encoding="utf-8")
    inventory = [{key: value for key, value in row.items() if key != "reward"} for row in rows]
    materialized_path_for(rollouts).write_text("".join(json.dumps(row) + "\n" for row in inventory), encoding="utf-8")
    return rollouts


@pytest.mark.parametrize("command", ["compare", "compare-suite"])
def test_real_cli_pools_sixteen_and_twenty_four_producer_repeats(
    command: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline_values = [[0.2 + 0.04 * run + 0.02 * repeat for repeat in range(8)] for run in range(2)]
    candidate_values = [[0.28 + 0.03 * run + 0.015 * repeat for repeat in range(8)] for run in range(3)]
    baseline = [_write_source(tmp_path / f"baseline {index}", values) for index, values in enumerate(baseline_values)]
    candidate = [
        _write_source(tmp_path / f"candidate {index}", values) for index, values in enumerate(candidate_values)
    ]
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    output = tmp_path / "reports"
    if command == "compare":
        options = ["--baseline-runs", ",".join(map(str, baseline)), "--candidate-runs", ",".join(map(str, candidate))]
    else:
        manifest = {
            "benchmarks": {
                "demo": {"metrics": [{"name": "mean/reward", "low": 0, "high": 1, "higher_is_better": True}]}
            },
            "models": {
                "model": {
                    "demo": {
                        "baseline": [{"rollouts": str(path)} for path in baseline],
                        "candidate": [{"rollouts": str(path)} for path in candidate],
                    }
                }
            },
        }
        suite = tmp_path / "suite.yaml"
        suite.write_text(yaml.safe_dump(manifest), encoding="utf-8")
        options = ["--suite", str(suite)]
    monkeypatch.setattr(
        "sys.argv", ["gym", "eval", command, *options, "--confidence-level", "0.95", "--output-dir", str(output)]
    )
    main()

    baseline_sample = [value for values in baseline_values for value in values]
    candidate_sample = [value for values in candidate_values for value in values]
    interval = ttest_ind(candidate_sample, baseline_sample, equal_var=False).confidence_interval(0.95)
    if command == "compare":
        report = json.loads((output / "compare_report.json").read_text())
        comparison = report["comparisons"][0]
        assert comparison["baseline_repeat_count"] == 16
        assert comparison["candidate_repeat_counts"] == [24]
        assert [source["rollouts_jsonl_fpath"] for source in report["baseline"]["sources"]] == list(map(str, baseline))
        assert [source["rollouts_jsonl_fpath"] for source in report["candidates"][0]["sources"]] == list(
            map(str, candidate)
        )
        row = next(row for row in comparison["metrics"] if row["metric"] == "mean/reward")
        estimate = row["candidates"][0]
        delta, low, high = estimate["delta"], estimate["delta_ci_low"], estimate["delta_ci_high"]
        scale = 1
        markdown = (output / "compare_report.md").read_text()
    else:
        report = json.loads((output / "suite_compare_report.json").read_text())
        cell = report["cells"][0]
        assert cell["baseline"]["repeat_count"] == 16
        assert cell["candidate"]["repeat_count"] == 24
        assert [source["rollouts"] for source in cell["baseline"]["sources"]] == list(map(str, baseline))
        assert [source["rollouts"] for source in cell["candidate"]["sources"]] == list(map(str, candidate))
        assert cell["baseline"]["coverage"]["status"] == "verified"
        assert cell["candidate"]["coverage"]["status"] == "verified"
        assert report["overall"]["weights"] == [1]
        estimate = report["overall"]["statistics"]
        delta, low, high = estimate["delta"], estimate["ci_low"], estimate["ci_high"]
        scale = 100
        markdown = (output / "suite_compare_report.md").read_text()
    assert delta == pytest.approx(scale * (mean(candidate_sample) - mean(baseline_sample)))
    assert low == pytest.approx(scale * interval.low)
    assert high == pytest.approx(scale * interval.high)
    for path in (*baseline, *candidate):
        assert str(path) in markdown
    assert all(path.read_bytes() == contents for path, contents in before.items())
