# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from scripts.harness_conformance.registry import P0_TOOL_TIMING_HARNESSES

from nemo_gym.rollouts.harness_capabilities.checker import EvidenceScope, inspect_record
from nemo_gym.rollouts.harness_capabilities.results import Results
from nemo_gym.rollouts.harness_capabilities.tool_timing import ToolTimingScope, check_tool_timing


def timing(boundary="lifecycle", source="test.hook"):
    return {"boundary": boundary, "timing_source": source, "duration_ms": 2.0, "duration_source": "monotonic"}


def check(tools, **kwargs):
    scope = ToolTimingScope(runtime_version="1.2.3", mode="sandbox", tool_paths=("bash",), **kwargs)
    results = Results()
    check_tool_timing(tools, scope, results, available=True)
    return {row["id"]: row["status"] for row in results.dump()}


def test_lifecycle_evidence_cannot_pass_executor_capability():
    rows = check(
        [{"tool_name": "bash", "timings": [timing()]}],
        lifecycle_sources=("test.hook",),
        executor_sources=("test.executor",),
    )
    assert rows["tools.timing.lifecycle"] == "pass"
    assert rows["tools.timing.executor"] == "fail"
    assert rows["tools.process_attribution"] == "not_applicable"


def test_missing_records_are_not_exempt_and_source_must_match():
    assert check([], lifecycle_sources=("test.hook",))["tools.timing.lifecycle"] == "fail"
    tools = [{"tool_name": "bash", "timings": [timing(source="another.hook")]}]
    assert check(tools, lifecycle_sources=("test.hook",))["tools.timing.lifecycle"] == "fail"


def test_duration_only_executor_and_zero_are_observations():
    measurement = timing("executor", "test.executor")
    measurement["duration_ms"] = 0
    rows = check(
        [{"tool_name": "bash", "timings": [measurement]}],
        executor_sources=("test.executor",),
        process_attribution=True,
    )
    assert rows["tools.timing.executor"] == "pass"
    assert rows["tools.process_attribution"] == "fail"
    measurement["duration_ms"] = None
    assert (
        check([{"tool_name": "bash", "timings": [measurement]}], executor_sources=("test.executor",))[
            "tools.timing.executor"
        ]
        == "fail"
    )


def test_scope_is_retained_in_existing_inspector_results():
    scope = ToolTimingScope(
        runtime_version="1.2.3", mode="sandbox", tool_paths=("bash",), lifecycle_sources=("test.hook",)
    )
    result = inspect_record(
        {"ng_trajectory": {"tool_calls": [{"tool_name": "bash", "timings": [timing()]}]}},
        scope=EvidenceScope(tool_timing=scope),
    )
    assert result["tool_timing_scope"]["runtime_version"] == "1.2.3"
    assert result["tool_timing_scope"]["mode"] == "sandbox"
    assert any(row["id"] == "tools.timing.lifecycle" and row["status"] == "pass" for row in result["checks"])


def test_scope_requires_runtime_mode_and_tool_paths():
    with pytest.raises(ValueError):
        ToolTimingScope(runtime_version="", mode="sandbox", tool_paths=("bash",))


def test_all_nine_timing_targets_resolve_real_adapter_components():
    root = Path(__file__).resolve().parents[4]
    assert len(P0_TOOL_TIMING_HARNESSES) == 9
    for component in P0_TOOL_TIMING_HARNESSES.values():
        assert (root / "responses_api_agents" / component / "app.py").is_file()
        assert (root / "responses_api_agents" / component / "configs" / f"{component}.yaml").is_file()


def test_public_scope_export_and_runner_configuration(tmp_path, capsys):
    import json

    from scripts.harness_conformance.registry import tool_timing_inventory
    from scripts.harness_conformance.runner import load_tool_timing_scopes, main

    from nemo_gym.rollouts.harness_capabilities import ToolTimingScope as PublicScope

    assert PublicScope is ToolTimingScope
    path = tmp_path / "scopes.json"
    path.write_text(
        json.dumps(
            {
                "opencode": {
                    "runtime_version": "1.2.3",
                    "mode": "local",
                    "tool_paths": ["bash"],
                    "lifecycle_sources": ["native"],
                }
            }
        )
    )
    assert load_tool_timing_scopes(path)["opencode"].lifecycle_sources == ["native"]
    assert main(["--list-timing-targets"]) == 0
    inventory = json.loads(capsys.readouterr().out)
    assert inventory == tool_timing_inventory()
    assert len(inventory) == 9
    assert inventory["nooa"]["runtime_probe_available"] is False
    assert all(row["timing_status"] == "unverified" for row in inventory.values())


def test_runner_passes_timing_expectations_to_artifact_inspection(tmp_path, monkeypatch):
    import json

    from scripts.harness_conformance import runner
    from scripts.harness_conformance.scenarios import SCENARIOS

    scenario = next(value for value in SCENARIOS if value.tool_steps > 0)
    (tmp_path / "rollouts.jsonl").write_text(
        json.dumps({"ng_trajectory": {"tool_calls": [{"tool_name": "bash", "timings": [timing()]}]}}) + "\n"
    )
    monkeypatch.setattr(runner, "inspect_behavior", lambda *args, **kwargs: [])
    monkeypatch.setattr(runner, "inspect_health", lambda *args, **kwargs: [])
    scope = ToolTimingScope(
        runtime_version="1.2.3",
        mode="local",
        tool_paths=("bash",),
        lifecycle_sources=("test.hook",),
        executor_sources=("executor",),
    )
    result = runner.inspect_episode(scenario, tmp_path, {"returncode": 0}, tool_timing=scope)
    checks = {check["id"]: check["status"] for check in result["checks"]}
    assert checks["tools.timing.lifecycle"] == "pass"
    assert checks["tools.timing.executor"] == "fail"
    assert result["tool_timing_scope"]["runtime_version"] == "1.2.3"
    assert result["verdict"] == "not_fulfilled"
