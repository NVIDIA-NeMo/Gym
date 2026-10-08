# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise wire protocols, independent witnesses, and fail-closed reporting."""

import copy
import json
import subprocess
import sys

import psutil
import pytest
from fastapi.testclient import TestClient
from scripts.harness_conformance.episode import HARNESSES, _config
from scripts.harness_conformance.provider import Probe
from scripts.harness_conformance.runner import _fingerprint, inspect_episode, main, run_process, run_suite
from scripts.harness_conformance.scenarios import SCENARIOS

from nemo_gym.base_responses_api_model import build_model_call_record
from nemo_gym.harness_capabilities.cli import json_rows
from tests.unit_tests.harness_capabilities.synthetic import evidence_and_witness


SCENARIO = {s.name: s for s in SCENARIOS}
TOOL = {
    "type": "function",
    "function": {
        "name": "bash",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}


def test_catalog_covers_p0_without_exposing_expectations():
    assert set().union(*(s.evidence for s in SCENARIOS)) == {f"TE-{i}" for i in range(1, 10)}
    for scenario in SCENARIOS:
        assert set(scenario.task()) == {"task_id", "responses_create_params"}
    assert main(["--list-scenarios"]) == 0


@pytest.mark.parametrize("harness", HARNESSES)
def test_launch_uses_local_endpoints_and_isolated_workspaces(harness, tmp_path):
    config = _config(harness, tmp_path, [10001, 10002, 10003, 10004], 12)
    from omegaconf import OmegaConf

    from nemo_gym.rollout_collection import _environment_server_for_agent, _environment_servers_by_agent

    assert (
        _environment_server_for_agent("probe_agent", _environment_servers_by_agent(OmegaConf.create(config)))
        == "probe_environment"
    )
    agent = config["probe_agent"]["responses_api_agents"][f"{harness}_agent"]
    assert agent["model_server"] == {"type": "responses_api_models", "name": "policy_model"}
    assert config["model_call_capture_dir"] == str(tmp_path / "capture")
    # Validate available adapters without invoking their auto-install hooks.
    if harness != "hermes":
        import importlib

        module = importlib.import_module(f"responses_api_agents.{harness}_agent.app")
        getattr(module, HARNESSES[harness] + "Config").model_validate(agent)


@pytest.mark.parametrize("scenario_name", ["tool_success", "tool_failure", "usage_omitted", "retry_429", "retry_500"])
def test_live_chat_endpoint_executes_script_and_captures_every_attempt(tmp_path, scenario_name):
    scenario = SCENARIO[scenario_name]
    probe = Probe(scenario, tmp_path)
    messages = [{"role": "user", "content": "Run checks"}]
    body = {"model": "conformance-model", "messages": messages, "tools": [TOOL]}
    with TestClient(probe.model_app()) as client:
        for status in scenario.http_errors:
            response = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
            assert response.status_code == status
        for index in range(2):
            response = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
            assert response.status_code == 200
            message = response.json()["choices"][0]["message"]
            call = message["tool_calls"][0]
            execution = subprocess.run(
                json.loads(call["function"]["arguments"])["command"],
                shell=True,
                capture_output=True,
                text=True,
                check=False,
            )
            assert execution.returncode == (scenario.tool_exit_code if index == 0 else 0)
            messages.extend([message, {"role": "tool", "tool_call_id": call["id"], "content": execution.stdout}])
        response = client.post("/ng-rollout/0-0/v1/chat/completions", json=body)
        assert response.json()["choices"][0]["message"]["content"] == "CONFORMANCE_DONE"
    assert probe.finished and not probe.violations
    assert all(t["executed"] and t["result_seen"] for t in probe.tool_calls)
    assert all(t["outputs"] == [t["token"]] for t in probe.tool_calls)
    captures = [
        build_model_call_record(row, call_index=i).model_dump()
        for i, row in json_rows(tmp_path / "capture/0-0.capture.jsonl")
    ]
    assert len(captures) == len(scenario.http_errors) + 3
    assert len({c["model_call_id"] for c in captures}) == len(captures)
    assert [c["status_code"] for c in captures] == [*scenario.http_errors, 200, 200, 200]
    if not scenario.usage:
        assert all(c.get("tokens_in") is None and c.get("tokens_out") is None for c in captures)
    else:
        assert captures[-1]["tokens_in"] == 20 + len(scenario.http_errors) + 2
        assert captures[-1]["tokens_reasoning"] == 2
        assert captures[-1]["cached_tokens"] == 3


@pytest.mark.parametrize("dialect", ["chat/completions", "responses"])
def test_streaming_tool_roundtrip_and_unknown_usage(tmp_path, dialect):
    probe = Probe(SCENARIO["usage_omitted"], tmp_path)
    body = {"model": "conformance-model", "stream": True}
    if dialect == "responses":
        body.update(
            input=[{"role": "user", "content": "Run checks"}],
            tools=[{"type": "namespace", "name": "functions", "tools": [{"type": "function", **TOOL["function"]}]}],
        )
    else:
        body.update(messages=[{"role": "user", "content": "Run checks"}], tools=[TOOL])
    with TestClient(probe.model_app()) as client:
        response = client.post(f"/ng-rollout/0-0/v1/{dialect}", json=body)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: {")]
    if dialect == "responses":
        assert events[-1]["type"] == "response.completed"
        call = events[-1]["response"]["output"][0]
        assert call["namespace"] == "functions" and call["name"] == "bash"
    else:
        assert response.text.endswith("data: [DONE]\n\n")
        assert any(e["choices"] and e["choices"][0]["delta"].get("tool_calls") for e in events)
    (capture,) = [
        build_model_call_record(row, call_index=i).model_dump()
        for i, row in json_rows(tmp_path / "capture/0-0.capture.jsonl")
    ]
    assert capture["tokens_in"] is None
    assert capture["tokens_out"] is None


def test_tool_request_without_execution_is_not_exercised(tmp_path):
    probe = Probe(SCENARIO["tool_success"], tmp_path)
    body = {"model": "probe", "messages": [], "tools": [TOOL]}
    with TestClient(probe.model_app()) as client:
        assert client.post("/v1/chat/completions", json=body).status_code == 200
        assert client.post("/v1/chat/completions", json=body).status_code == 409
    assert probe.violations and not probe.finished


def test_terminal_failure_does_not_turn_into_a_success_on_retry(tmp_path):
    probe = Probe(SCENARIO["model_error"], tmp_path)
    with TestClient(probe.model_app()) as client:
        for _ in range(3):
            response = client.post("/v1/responses", json={"model": "probe", "input": []})
            assert response.status_code == 400
            assert "id" not in response.json()
    assert len(probe.attempts) == 3 and not probe.finished


@pytest.mark.parametrize("name,reward", [("tool_success", 1.0), ("verifier_failure", 0.0)])
def test_verifier_records_actual_final_answer(tmp_path, name, reward):
    probe = Probe(SCENARIO[name], tmp_path)
    body = {
        "response": {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "CONFORMANCE_DONE"}],
                }
            ]
        }
    }
    with TestClient(probe.resources_app()) as client:
        client.post("/seed_session", json={})
        response = client.post("/ng-rollout/0-0/verify", json=body)
    assert response.json()["reward"] == reward
    assert response.json()["evaluation_completed"] is True
    assert response.json()["mask_sample"] is False
    assert probe.verifications == [{"reward": reward, "answer_seen": True}]
    assert probe.seeded == 1


@pytest.fixture
def retained_episode(tmp_path):
    # Contract fixture only: used to prove a green artifact cannot hide a missing live probe.
    record, witness = evidence_and_witness()
    calls = [
        {**c, "status_code": c["response_metadata"]["status_code"]} for c in record["ng_trajectory"]["model_calls"]
    ]
    (tmp_path / "rollouts.jsonl").write_text(json.dumps(record) + "\n")
    (tmp_path / "witness.json").write_text(json.dumps(witness))
    (tmp_path / "capture").mkdir()
    (tmp_path / "capture/0-0.capture.jsonl").write_text("".join(json.dumps(c) + "\n" for c in calls))
    return tmp_path, witness


def test_behavior_failure_does_not_erase_artifact_success(retained_episode):
    directory, witness = retained_episode
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert result["verdict"] == "fulfilled", result["issues"]
    witness["attempts"].append(copy.deepcopy(witness["attempts"][0]))
    (directory / "witness.json").write_text(json.dumps(witness))
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert result["verdict"] == "not_fulfilled"
    assert result["behavioral_status"] == "fail"
    assert all(v["verdict"] == "fulfilled" for v in result["evidence"].values())
    assert "canonical" in " ".join(result["issues"])


def test_execution_status_does_not_erase_retained_results(retained_episode):
    directory, _ = retained_episode
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 1, "timed_out": True})
    assert result["verdict"] == "fulfilled" and result["behavioral_status"] == "pass"
    assert result["execution"] == {"returncode": 1, "timed_out": True}
    assert not any(c["id"].startswith("execution") for c in result["checks"])


@pytest.mark.parametrize(
    "mutation", ["missing", "duplicate", "extra", "arguments", "name", "output", "status", "owner"]
)
def test_tool_witness_rejects_lost_or_changed_evidence(retained_episode, mutation):
    directory, _ = retained_episode
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    trajectory = record["ng_trajectory"]
    tools = trajectory["tool_calls"]
    invocations = [trajectory["invocations"][0], record["ng_agent_observations"]["records"][0]]
    if mutation == "missing":
        tools.pop()
        record["ng_agent_observations"]["records"].pop()
        for invocation in invocations:
            invocation["conversation"] = [i for i in invocation["conversation"] if i.get("call_id") != "tool-2"]
    elif mutation in {"duplicate", "extra"}:
        tool = copy.deepcopy(tools[0])
        if mutation == "extra":
            tool["tool_call_id"] = "unwitnessed-tool"
        tools.append(tool)
    elif mutation in {"arguments", "name"}:
        for invocation in invocations:
            invocation["conversation"][1][mutation] = '{"key":"changed"}' if mutation == "arguments" else "other_tool"
        if mutation == "name":
            tools[0]["tool_name"] = "other_tool"
    elif mutation == "output":
        tools[0]["output"] = "changed"
        for invocation in invocations:
            invocation["conversation"][2]["output"] = "changed"
    elif mutation == "status":
        tools[0]["status"] = "failed"
    else:
        tools[0]["invocation_id"] = "other-invocation"
    bundle.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert result["verdict"] == "not_fulfilled" and result["behavioral_status"] == "fail"
    assert any("retained tool" in issue for issue in result["issues"])
    assert not any("retained model attempts differ" in issue for issue in result["issues"])
    if mutation in {"missing", "arguments", "name", "output", "status"}:
        # These artifacts remain internally consistent; the independent witness
        # must expose the lost/changed evidence even when TE-5 alone passes.
        assert result["evidence"]["TE-5"]["artifact_verdict"] == "fulfilled"


@pytest.mark.parametrize("surface", ["trajectory", "observations"])
def test_tool_witness_uses_only_designated_surface(retained_episode, surface):
    directory, _ = retained_episode
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    if surface == "trajectory":
        del record["ng_agent_observations"]
    else:
        del record["ng_trajectory"]["tool_calls"]
        del record["ng_trajectory"]["invocations"]
    bundle.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert bool(result["issues"]) == (surface == "observations")
    assert result["verdict"] == ("fulfilled" if surface == "trajectory" else "not_fulfilled")


@pytest.mark.parametrize("failed", [True, False])
def test_tool_witness_requires_prescribed_failure_status(retained_episode, failed):
    directory, witness = retained_episode
    witness["tool_calls"][0]["exit_code"] = 7
    (directory / "witness.json").write_text(json.dumps(witness))
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    if failed:
        record["ng_trajectory"]["tool_calls"][0]["status"] = "failed"
    bundle.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert (result["behavioral_status"] == "pass") == failed
    assert ("retained tool status differs from the independent tool witness" in result["issues"]) != failed


def test_missing_runtime_is_recorded_and_no_stale_output_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", "")
    output = tmp_path / "run"
    summary, code = run_suite(harnesses=["pi", "opencode"], scenarios=(SCENARIOS[0],), output=output, timeout=1)
    assert code == 2 and not summary["full_suite"]
    for row in summary["harnesses"].values():
        assert row["evidence"]["TE-1"] == {"required": 1, "observed": 0, "passed": 0}
    assert (output / "conformance_summary.json").is_file()
    before = (output / "suite.json").read_bytes()
    manifest = json.loads(before)
    assert {"behavior.py", "checker.py", "checks.py"} <= manifest["checker_sources"].keys()
    assert "behavior.py" not in manifest["runner_sources"]
    assert main(["--harness", "pi", "--output", str(output)]) == 2
    assert (output / "suite.json").read_bytes() == before


def test_timeout_reaps_descendant_in_another_process_group(tmp_path):
    child_code = "import time; time.sleep(300)"
    worker = (
        "import subprocess,sys,time; from pathlib import Path; "
        f"p=subprocess.Popen([sys.executable,'-c',{child_code!r}],start_new_session=True); "
        f"Path({str(tmp_path / 'pid')!r}).write_text(str(p.pid)); time.sleep(300)"
    )
    result = run_process([sys.executable, "-c", worker], directory=tmp_path, timeout=1)
    assert result["timed_out"]
    pid = int((tmp_path / "pid").read_text())
    assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_invalid_timeout_rejected_before_launch(tmp_path, timeout):
    with pytest.raises(SystemExit) as exc:
        main(["--output", str(tmp_path / "absent"), "--timeout", timeout])
    assert exc.value.code == 2
    assert not (tmp_path / "absent").exists()


def test_witness_detects_changed_payload_even_when_attempt_identity_matches(retained_episode):
    directory, witness = retained_episode
    witness["attempts"][0]["response"]["usage"]["total_tokens"] += 1
    (directory / "witness.json").write_text(json.dumps(witness))
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert result["verdict"] == "not_fulfilled"
    assert "canonical" in " ".join(result["issues"])


def test_checker_error_does_not_prevent_remaining_episodes(tmp_path, monkeypatch):
    from scripts.harness_conformance import runner

    inspected = []
    monkeypatch.setattr(runner, "run_process", lambda *args, **kwargs: {"returncode": 0, "timed_out": False})

    def fail_inspection(scenario, directory, execution):
        inspected.append(scenario.name)
        raise ValueError("truncated rollout")

    monkeypatch.setattr(runner, "inspect_episode", fail_inspection)
    summary, code = run_suite(harnesses=["pi"], scenarios=SCENARIOS[:2], output=tmp_path / "run", timeout=1)
    assert code == 2
    assert inspected == ["tool_success", "tool_failure"]
    assert summary["harnesses"]["pi"]["evidence"]["TE-1"] == {"required": 2, "observed": 0, "passed": 0}


def test_interrupted_run_does_not_publish_completed_summary(tmp_path, monkeypatch):
    from scripts.harness_conformance import runner

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner, "run_process", interrupt)
    output = tmp_path / "run"
    with pytest.raises(KeyboardInterrupt):
        run_suite(harnesses=["pi"], scenarios=SCENARIOS[:1], output=output, timeout=1)
    assert not (output / "conformance_summary.json").exists()


@pytest.mark.parametrize(
    "collection,field,value",
    [("turns", "resolved", "wrong"), ("model_calls", "response_metadata", False), ("tool_calls", "tool_call_id", [])],
)
def test_malformed_evidence_keeps_other_check_results(retained_episode, collection, field, value):
    directory, _ = retained_episode
    path = directory / "rollouts.jsonl"
    record = json.loads(path.read_text())
    record["ng_trajectory"][collection][0][field] = value
    path.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    checks = {c["id"]: c for c in result["checks"]}
    assert any(c["status"] == "fail" for c in checks.values())
    assert checks["tokens.prompt_tokens"]["status"] == "pass"


def test_missing_witness_does_not_fail_retained_evidence(retained_episode):
    directory, _ = retained_episode
    (directory / "witness.json").unlink()
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert all(c["status"] in {"fail", "not_applicable"} for c in result["checks"] if c["kind"] == "behavioral")
    assert all(v["verdict"] == "fulfilled" for v in result["evidence"].values())


@pytest.mark.parametrize("tier,verdict", [("P0", "not_fulfilled"), ("P1", "fulfilled"), ("P2", "fulfilled")])
def test_scenario_gate_uses_individual_priority(retained_episode, monkeypatch, tier, verdict):
    from dataclasses import replace

    from scripts.harness_conformance import runner

    directory, _ = retained_episode
    original = runner.inspect_behavior

    def with_failing_check(*args, **kwargs):
        checks = original(*args, **kwargs)
        # No TE label: this rule's priority alone determines whether it blocks P0.
        checks.append(
            {
                "id": "independent.test",
                "kind": "behavioral",
                "tier": tier,
                "status": "fail",
                "evidence": (),
                "locations": [],
                "reasons": ["mismatch"],
                "blocked_by": [],
            }
        )
        return checks

    monkeypatch.setattr(runner, "inspect_behavior", with_failing_check)
    scenario = replace(SCENARIO["verifier_failure"], evidence=())
    result = inspect_episode(scenario, directory, {"returncode": 0, "timed_out": False})
    assert result["verdict"] == verdict
    assert result["behavioral_status"] == "fail"
    assert result["evidence"] == {}
    assert all(c["status"] in {"pass", "not_applicable"} for c in result["checks"] if c["id"] != "independent.test")


def test_step_attribution_failure_blocks_scenario_p0(retained_episode):
    directory, _ = retained_episode
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    for turn in record["ng_trajectory"]["turns"]:
        turn["model_calls"] = []
    bundle.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO["verifier_failure"], directory, {"returncode": 0, "timed_out": False})
    assert result["evidence"]["TE-8"]["verdict"] == "fulfilled"
    assert result["evidence"]["TE-9"]["verdict"] == "not_fulfilled"
    assert result["verdict"] == "not_fulfilled"


@pytest.mark.parametrize("delivery", ["missing", "failure_record"])
def test_missing_rollout_fails_required_checks_and_preserves_exclusions(tmp_path, delivery):
    if delivery == "failure_record":
        (tmp_path / "rollouts_failures.jsonl").write_text('{"failure": "model error"}\n')
    result = inspect_episode(SCENARIO["model_error"], tmp_path, {"returncode": 1, "timed_out": False})
    checks = {c["id"]: c for c in result["checks"]}
    assert checks["calls.outcome"]["status"] == "fail"
    assert checks["calls.outcome"]["reasons"][0].startswith("required input is unavailable")
    assert checks["tool_calls.present"]["status"] == "not_applicable"
    assert checks["evaluation.reward"]["status"] == "not_applicable"
    assert checks["turns.present"]["status"] == "not_applicable"
    assert checks["steps.attempt_accounting"]["status"] == "not_applicable"
    assert result["evidence"]["TE-1"]["verdict"] == "not_fulfilled"
    assert result["behavioral_status"] == "fail"
    assert result["delivery"] == delivery
    assert result["execution"]["returncode"] == 1


@pytest.mark.parametrize("scenario_name", ["model_error", "retry_429", "retry_500"])
def test_step_scope_is_declared_by_scenario_not_missing_turns(retained_episode, scenario_name):
    directory, _ = retained_episode
    bundle = directory / "rollouts.jsonl"
    record = json.loads(bundle.read_text())
    record["ng_trajectory"]["turns"] = []
    bundle.write_text(json.dumps(record) + "\n")
    result = inspect_episode(SCENARIO[scenario_name], directory, {"returncode": 0, "timed_out": False})
    checks = {c["id"]: c for c in result["checks"]}
    expected = "not_applicable" if scenario_name == "model_error" else "fail"
    for check_id in ("turns.present", "steps.number", "steps.call_owner", "steps.attempt_accounting"):
        assert checks[check_id]["status"] == expected
    for check_id in ("calls.identity", "calls.timing", "calls.outcome", "ownership.call_owner"):
        assert checks[check_id]["status"] != "not_applicable"


def test_chat_fingerprint_ignores_creation_clock():
    request = {"messages": [{"role": "user", "content": "hello"}]}
    response = {"id": "r", "object": "chat.completion", "created": 123, "choices": []}
    without_clock = {key: value for key, value in response.items() if key != "created"}
    assert _fingerprint(request, 200, response) == _fingerprint(request, 200, without_clock)
