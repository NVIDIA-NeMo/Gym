# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys

import pytest

from nemo_gym.standard_analyses import analyze
from nemo_gym.standard_analyses.dashboard import build_workspace, chart_specifications
from nemo_gym.standard_analyses.export import log_wandb, write_report
from nemo_gym.standard_analyses.inputs import from_gym, read_records
from tests.unit_tests.standard_analyses.fixtures import demo_records, gym_rows


def gym(row):
    return from_gym(row, run_id="run")[0]


def result(report, name):
    return next(r for r in report.results if r.name == name)


def test_gym_trajectory_adapter_matches_call_measurements():
    canonical, adapted = demo_records()[0], gym(gym_rows()[0])
    assert adapted.calls == canonical.calls
    assert adapted.capture_complete is True


def test_gym_actual_index_keys_and_missing_reward():
    row = gym_rows()[0]
    row["_ng_task_index"] = 42
    row.pop("reward")
    r = gym(row)
    assert r.task_id == "42" and r.repeat == 0 and r.success is None


def test_capture_gaps_disable_repeat_comparison_and_nonbinary_reward_is_unknown():
    row = gym_rows()[0]
    row["ng_model_call_capture"]["gaps"] = [{"code": "model_call_capture_incomplete"}]
    row["reward"] = 0.5
    r = gym(row)
    assert r.capture_complete is False and r.success is None


def test_raw_capture_does_not_override_or_duplicate_canonical_measurements():
    row = gym_rows()[0]
    row["ng_model_call_capture"]["calls"][0]["tokens_out"] = 999999
    assert gym(row).calls[0].output_tokens == 120
    assert len(gym(row).calls) == 6


def test_trajectory_only_capture_is_not_assumed_complete():
    row = gym_rows()[0]
    row.pop("ng_model_call_capture")
    assert gym(row).capture_complete is None


def test_invalid_trajectory_is_unavailable_instead_of_aborting_collection():
    row = gym_rows()[0]
    row["ng_trajectory"]["schema_version"] = "invalid"
    r = gym(row)
    assert not r.calls
    assert r.capture_complete is False
    assert "trajectory_invalid" in r.gaps


def test_invalid_call_measurements_remain_in_the_denominator():
    row = gym_rows()[0]
    row["ng_trajectory"]["model_calls"][0]["token_stats"]["reasoning_tokens"] = 10000
    r = gym(row)
    assert len(r.calls) == 6
    assert r.calls[0].output_tokens is None
    assert not r.capture_complete


def test_payloads_commands_and_file_paths_never_reach_results():
    row = gym_rows()[0]
    row["ng_trajectory"]["model_calls"][0]["request"] = {"secret": "DO_NOT_EXPORT"}
    serialized = json.dumps(analyze([gym(row)]).to_dict())
    assert "DO_NOT_EXPORT" not in serialized


def test_local_report_and_multifile_shard_merging(tmp_path):
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in gym_rows()))
    groups = read_records([path, path], run_id="run")
    assert len(groups) == 1
    report = analyze(next(iter(groups.values())))
    write_report(report, tmp_path / "merged")
    data = json.loads((tmp_path / "merged/results.json").read_text())
    assert data["results"][0]["coverage"]["rollouts_total"] == 12


def test_core_runs_without_importing_tracking_sdks():
    code = "import sys; from tests.unit_tests.standard_analyses.fixtures import demo_records; from nemo_gym.standard_analyses import analyze; analyze(demo_records()); assert 'wandb' not in sys.modules; assert 'wandb_workspaces' not in sys.modules"
    subprocess.run([sys.executable, "-c", code], check=True)


def test_wandb_sdk_serializes_typed_tables_charts_and_workspace_offline(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    pytest.importorskip("wandb_workspaces")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("WANDB_CONFIG_DIR", str(tmp_path / "config"))
    report = analyze(demo_records())
    with wandb.init(
        project="standard-analyses-test", mode="offline", dir=str(tmp_path), config={"standard_analyses": True}
    ) as run:
        logged = []
        original_log = run.log

        def capture_log(payload):
            logged.append(dict(payload))
            original_log(payload)

        monkeypatch.setattr(run, "log", capture_log)
        log_wandb(report, run)
        for key, value in report.scalars().items():
            assert logged[0][key] == value
        assert run.summary["efficiency/long_calls/output_gt_1000_time_fraction"] > 0
        assert run.summary["efficiency/availability"]["nrows"] == 5
        assert run.summary["efficiency/repeat_excess/detail"]["nrows"] == 21
        log_wandb(report, run, namespace="efficiency/cohorts/example")
        assert all(key.startswith("efficiency/cohorts/example/") for key in logged[1])
        assert run.summary["efficiency/cohorts/example/repeat_excess/detail"]["nrows"] == 21
    view = build_workspace("test-entity", "test-project")
    payload = view._to_model().model_dump_json(by_alias=True)
    assert "efficiency/repeat_excess/detail" in payload
    assert "Standard analyses" in payload
    # ScalarChart reads history; SummaryMetric would introduce an invalid
    # summary_metrics. prefix that leaves the published panels empty.
    for panel in view.sections[0].panels:
        metric = panel._to_model().config.metrics[0]
        assert metric in logged[0]
        assert not metric.startswith("summary_metrics.")
    charts = chart_specifications()
    assert set(charts) == {result.name for result in report.results}
    for section, result in zip(view.sections[2:], report.results, strict=True):
        assert section.is_open
        assert len(section.panels) == len(charts[result.name]) + 1
        for spec, panel in zip(charts[result.name], section.panels[:-1], strict=True):
            model = panel._to_model()
            if "metrics" in spec:
                assert all(key in report.scalars() for key in model.config.metrics)
            else:
                key = "efficiency/" + spec["table"]
                table = logged[0][key]
                assert set(spec["fields"].values()) <= set(table.columns)
                query = model.config.user_query.query_fields[0]
                table_field = next(field for field in query.fields if field.name == "summaryTable")
                assert table_field.args[0].value == key
                assert model.config.panel_def_id == f"wandb/{spec['preset']}/v0"


def test_masked_rollout_is_excluded_from_repeat_scores():
    rows = gym_rows()
    rows[0]["mask_sample"] = True
    records = [gym(row) for row in rows]
    assert not records[0].eligible
    report = analyze(records)
    assert all(result.coverage["rollouts_excluded_health"] == 1 for result in report.results)
