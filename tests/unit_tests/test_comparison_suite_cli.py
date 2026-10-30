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
"""Exercise the suite comparison command and real rendered artifacts."""

import json
from pathlib import Path

import pytest
import yaml
from hydra.core.override_parser.overrides_parser import OverridesParser

import nemo_gym.cli.eval as cli_eval
from nemo_gym.cli.main import build_parser, main
from nemo_gym.comparison.suite.report import render_suite_markdown, render_suite_summary
from nemo_gym.comparison.suite.schema import SuiteComparisonResult


def _saved_run(directory: Path, values: list[float]) -> str:
    directory.mkdir()
    payload = [
        {
            "agent_ref": {"name": "agent"},
            "agent_metrics": {"num_repeats": len(values)},
            "group_level_metrics": [
                {
                    "_ng_task_index": task,
                    "rollout_infos": [
                        {"_ng_rollout_index": repeat, "reward": value} for repeat, value in enumerate(values)
                    ],
                }
                for task in range(2)
            ],
            "repeat_level_metrics": [
                {"_ng_rollout_index": repeat, "sample_count": 2, "mean/reward": value}
                for repeat, value in enumerate(values)
            ],
        }
    ]
    (directory / "rollouts_aggregate_metrics.json").write_text(json.dumps(payload))
    return str(directory / "rollouts.jsonl")


@pytest.fixture
def suite_path(tmp_path: Path) -> Path:
    def pair(name: str, baseline: list[float], candidate: list[float]) -> dict:
        return {
            "baseline": {"rollouts": _saved_run(tmp_path / f"{name}-base", baseline)},
            "candidate": {"rollouts": _saved_run(tmp_path / f"{name}-candidate", candidate)},
        }

    payload = {
        "schema_version": 1,
        "benchmarks": {
            name: {"metrics": [{"name": "mean/reward", "low": 0, "high": 1, "higher_is_better": True}]}
            for name in ("first", "second")
        },
        "models": {
            "model-a": {
                "first": pair("first", [0.2, 0.4, 0.6], [0.3, 0.5, 0.7, 0.9]),
                "second": pair("second", [0.4, 0.6, 0.8], [0.5, 0.7, 0.9]),
            }
        },
    }
    path = tmp_path / "suite with spaces.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return path


def test_suite_flags_survive_the_hydra_translation() -> None:
    args, unknown = build_parser().parse_known_args(
        ["eval", "compare-suite", "--suite", "results/[v2] suite.yaml", "--confidence-level", "0.95"]
    )
    assert unknown == []
    tokens = [token for flag in args._command.flags for token in flag.translate_to_hydra(args)]
    parsed = OverridesParser.create().parse_overrides(tokens)
    values = {override.key_or_group: override.value() for override in parsed}
    assert values == {
        "suite_fpath": "results/[v2] suite.yaml",
        "confidence_level": 0.95,
    }


def test_missing_suite_reports_the_required_flag(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli_eval, "get_global_config_dict", lambda: {})
    with pytest.raises(SystemExit) as error:
        cli_eval.compare_suite()
    assert error.value.code == 1
    assert "gym eval compare-suite requires --suite PATH" in " ".join(capsys.readouterr().out.split())


def test_real_cli_writes_numeric_reports_without_overwriting_run_reports(
    suite_path: Path, monkeypatch, capsys
) -> None:
    output = suite_path.parent / "reports"
    output.mkdir()
    existing = output / "compare_report.md"
    existing.write_text("existing run report")
    monkeypatch.setattr(
        "sys.argv",
        [
            "gym",
            "eval",
            "compare-suite",
            "--suite",
            str(suite_path),
            "--output-dir",
            str(output),
            "++policy_api_key=sk-suite-example",
        ],
    )
    main()
    report_bytes = (output / "suite_compare_report.json").read_bytes()
    result = SuiteComparisonResult.model_validate_json(report_bytes)
    assert result.overall.statistics.delta == pytest.approx(15)
    assert result.overall.statistics.baseline_mean == pytest.approx(50)
    assert result.overall.statistics.candidate_mean == pytest.approx(65)
    payload = json.loads(report_bytes)
    for comparison in [*payload["per_model"], payload["overall"], *payload["cells"]]:
        statistics = comparison["statistics"]
        assert statistics["delta"] == pytest.approx(statistics["candidate_mean"] - statistics["baseline_mean"])
        assert "gap" not in statistics
        assert all(key.isascii() for key in statistics)
    assert result.confidence_level == 0.9
    assert result.overall.weights == [0.5, 0.5]
    assert result.command.startswith("gym eval compare-suite ")
    assert "comparison_level" not in result.command
    assert "++policy_api_key=****" in result.command
    assert "sk-suite-example" not in (output / "suite_compare_report.json").read_text()
    markdown = (output / "suite_compare_report.md").read_text()
    assert markdown.startswith("# gym eval compare-suite\n")
    assert "sk-suite-example" not in markdown
    assert "| Aggregate | Cells | Candidate | Baseline | Delta | 90% CI for delta |" in markdown
    assert "| model-a | 2 | 65.00 | 50.00 | 15.00 |" in markdown
    assert "| Model | Benchmark | Status | Candidate repeats | Baseline repeats | Delta |" in markdown
    assert "| model-a | first | included | 4 | 3 | 20.00 |" in markdown
    assert "Delta = candidate − baseline; positive deltas favor the candidate." in markdown
    assert "gap" not in markdown.lower()
    assert "95% CI for delta" not in markdown
    assert "Planned coverage is unverified" in markdown
    assert existing.read_text() == "existing run report"
    stdout = capsys.readouterr().out
    assert "model-a: candidate=65.00 baseline=50.00 delta=15.00" in stdout


def test_markdown_uses_four_significant_digits_without_rounding_json(tmp_path: Path, monkeypatch) -> None:
    samples = {
        "tiny": ([0.1234567, 0.1234568, 0.1234569], [0.12345672, 0.12345682, 0.12345692]),
        "ordinary": ([0.123456, 0.234567, 0.345678], [0.111111, 0.222222, 0.333333]),
        "constant": ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    }
    manifest = {
        "benchmarks": {
            name: {
                "metrics": [
                    {
                        "name": "mean/reward",
                        "low": -123456.789 if name == "constant" else 0,
                        "high": 987654.321 if name == "constant" else 1,
                        "higher_is_better": True,
                    }
                ]
            }
            for name in samples
        },
        "models": {
            "model": {
                name: {
                    "baseline": {"rollouts": _saved_run(tmp_path / f"{name}-baseline", baseline)},
                    "candidate": {"rollouts": _saved_run(tmp_path / f"{name}-candidate", candidate)},
                }
                for name, (baseline, candidate) in samples.items()
            }
        },
    }
    suite = tmp_path / "suite.yaml"
    suite.write_text(yaml.safe_dump(manifest, sort_keys=False))
    output = tmp_path / "reports"
    monkeypatch.setattr(
        "sys.argv", ["gym", "eval", "compare-suite", "--suite", str(suite), "--output-dir", str(output)]
    )
    main()

    markdown = (output / "suite_compare_report.md").read_text()
    assert "| model | 3 | 15.23 | 15.64 | -0.4115 | [-6.858, 6.035] | 3.024 | 4.000 |" in markdown
    assert "| model | tiny | included | 3 | 3 | 2.000e-06 | [-1.541e-05, 1.941e-05] | 0.3333 | 0.3333 |" in markdown
    assert "| model | constant | included | 3 | 3 | 0.000 | unavailable | 0.3333 | 0.3333 |" in markdown
    assert "| constant | mean/reward | -1.235e+05 | 9.877e+05 | True |" in markdown

    payload = json.loads((output / "suite_compare_report.json").read_bytes())
    assert payload["overall"]["statistics"]["candidate_mean"] == pytest.approx(15.226346007037051)
    assert payload["overall"]["statistics"]["candidate_mean"] != 15.23
    assert payload["cells"][0]["statistics"]["delta"] == pytest.approx(2e-6)
    assert payload["cells"][0]["statistics"]["ci_low"] == pytest.approx(-1.5406456118259828e-5)
    assert payload["cells"][0]["model_weight"] == 1 / 3
    assert payload["cells"][2]["metrics"][0]["high"] == 987654.321


def test_normalization_table_lists_each_benchmark_metric_once_across_models(suite_path: Path, monkeypatch) -> None:
    manifest = yaml.safe_load(suite_path.read_text())
    manifest["benchmarks"]["first"]["metrics"].append(
        {"name": "mean/latency", "low": 0, "high": 10, "higher_is_better": False}
    )
    manifest["benchmarks"]["second"]["metrics"][0].update(high=100, higher_is_better=False)
    manifest["models"]["model-b"] = {
        benchmark: {
            role: {"rollouts": _saved_run(suite_path.parent / f"model-b-{benchmark}-{role}", values)}
            for role, values in (("baseline", [0.1, 0.3, 0.5]), ("candidate", [0.4, 0.6, 0.8]))
        }
        for benchmark in ("first", "second")
    }
    for benchmarks in manifest["models"].values():
        for source in benchmarks["first"].values():
            aggregate_path = Path(source["rollouts"]).with_name("rollouts_aggregate_metrics.json")
            aggregate = json.loads(aggregate_path.read_bytes())
            for repeat, metrics in enumerate(aggregate[0]["repeat_level_metrics"]):
                metrics["mean/latency"] = 1.5 + repeat
            aggregate_path.write_text(json.dumps(aggregate))
    suite_path.write_text(yaml.safe_dump(manifest, sort_keys=False))
    output = suite_path.parent / "reports"
    monkeypatch.setattr(
        "sys.argv", ["gym", "eval", "compare-suite", "--suite", str(suite_path), "--output-dir", str(output)]
    )
    main()

    markdown = (output / "suite_compare_report.md").read_text()
    normalization = markdown.split("## Metric normalization\n", 1)[1].split("\n## ", 1)[0]
    assert [line for line in normalization.splitlines() if line.startswith("|")] == [
        "| Benchmark | Metric | Theoretical low | Theoretical high | Higher is better | Definition source |",
        "| --- | --- | ---: | ---: | --- | --- |",
        "| first | mean/reward | 0.000 | 1.000 | True | suite manifest override |",
        "| first | mean/latency | 0.000 | 10.00 | False | suite manifest override |",
        "| second | mean/reward | 0.000 | 100.0 | False | suite manifest override |",
    ]
    assert "Model" not in normalization
    result = SuiteComparisonResult.model_validate_json((output / "suite_compare_report.json").read_bytes())
    assert [(cell.model, cell.benchmark) for cell in result.cells] == [
        ("model-a", "first"),
        ("model-a", "second"),
        ("model-b", "first"),
        ("model-b", "second"),
    ]
    assert result.cells[0].metrics[0].baseline_raw != result.cells[2].metrics[0].baseline_raw


def test_adjustable_confidence_and_json_only_output(suite_path: Path, monkeypatch) -> None:
    output = suite_path.parent / "json-only"
    monkeypatch.setattr(
        "sys.argv",
        [
            "gym",
            "eval",
            "compare-suite",
            "--suite",
            str(suite_path),
            "--output-dir",
            str(output),
            "--confidence-level",
            "0.955",
            "--report-format",
            "json",
        ],
    )
    main()
    result = SuiteComparisonResult.model_validate_json((output / "suite_compare_report.json").read_bytes())
    assert result.confidence_level == 0.955
    assert result.overall.statistics.confidence_level == 0.955
    assert "95.5% CI for delta" in render_suite_markdown(result)
    assert "95.5% confidence" in render_suite_summary(result)
    assert not (output / "suite_compare_report.md").exists()
    assert result.overall.statistics.ci_low < result.overall.statistics.delta < result.overall.statistics.ci_high


def test_report_destination_cannot_overwrite_an_input(suite_path: Path, monkeypatch, capsys) -> None:
    payload = yaml.safe_load(suite_path.read_text())
    baseline = payload["models"]["model-a"]["first"]["baseline"]
    original = Path(baseline["rollouts"]).with_name("rollouts_aggregate_metrics.json")
    output = suite_path.parent / "collision"
    output.mkdir()
    collision = output / "suite_compare_report.json"
    original_bytes = original.read_bytes()
    collision.write_bytes(original_bytes)
    baseline["aggregate_metrics"] = str(collision)
    suite_path.write_text(yaml.safe_dump(payload))
    monkeypatch.setattr(
        "sys.argv",
        [
            "gym",
            "eval",
            "compare-suite",
            "--suite",
            str(suite_path),
            "--output-dir",
            str(output),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 1
    assert "would overwrite a source artifact" in " ".join(capsys.readouterr().out.split())
    assert collision.read_bytes() == original_bytes
    assert not (output / "suite_compare_report.md").exists()
