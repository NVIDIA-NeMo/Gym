# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Conformance expectations exercised through Gym's real persisted-artifact health engine."""

import json
from copy import deepcopy

import pytest
from scripts.harness_conformance.scenarios import SCENARIOS

from nemo_gym.harness_capabilities import health
from nemo_gym.harness_capabilities.results import gate_passes, render_matrices
from nemo_gym.health.types import CheckScope, CheckSpec, CheckSubject
from tests.unit_tests.harness_capabilities.synthetic import evidence_record


@pytest.fixture
def record():
    record = evidence_record()
    record["response"]["usage"] = {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}
    return record


def inspect(tmp_path, records, expectations=None, **kwargs):
    path = tmp_path / "rollouts.jsonl"
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    checks = health.inspect_health([path], output=tmp_path / "health", expectations=expectations or {}, **kwargs)
    return {row["id"].removeprefix("health."): row for row in checks}


def test_default_health_checks_use_real_engine_and_publish_reports(tmp_path, record, monkeypatch):
    original = health.run_health_checks
    calls = []

    def spy(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(health, "run_health_checks", spy)
    checks = inspect(tmp_path, [record])
    assert len(calls) == 1
    assert set(checks) == {s.id for s in health.CHECK_REGISTRY if s.evaluation_scope == CheckScope.ROLLOUT}
    assert all(c["status"] == "pass" and c["tier"] == "P0" for c in checks.values())
    assert (tmp_path / "health/quality_summary.json").is_file()
    assert json.loads((tmp_path / "health/rollout_verdicts.jsonl").read_text())["verdict"] == "healthy"
    table = render_matrices({"test": {"scenarios": [{"scenario": "success", "checks": list(checks.values())}]}})
    assert "**Health**" in table and "`health.rollout_token_count_mismatch` | ✓" in table


def test_missing_usage_only_passes_explicit_scenario_expectations(tmp_path, record):
    for call in record["ng_trajectory"]["model_calls"]:
        call["token_stats"] = {}
    del record["response"]["usage"]
    checks = inspect(tmp_path, [record])
    assert checks["model_call_missing_token_counts"]["status"] == "fail"
    assert checks["rollout_token_count_mismatch"]["reasons"] == ["expected healthy; observed unobserved"]
    checks = inspect(
        tmp_path,
        [record],
        {"model_call_missing_token_counts": "unhealthy", "rollout_token_count_mismatch": "unobserved"},
    )
    assert gate_passes(list(checks.values()))
    assert json.loads((tmp_path / "health/rollout_verdicts.jsonl").read_text())["verdict"] == "unhealthy"
    # Expected unobserved for tokens must not excuse an unrelated loss of evidence.
    record["ng_trajectory"]["gaps"] = [{"code": "turns_unavailable"}]
    checks = inspect(tmp_path, [record], {"rollout_token_count_mismatch": "unobserved"})
    assert checks["agent_turn_hollow"]["status"] == "fail"


def test_aggregate_usage_mismatch_fails_without_mutating_artifacts(tmp_path, record):
    record["response"]["usage"]["input_tokens"] = 999
    checks = inspect(tmp_path, [record])
    assert checks["rollout_token_count_mismatch"]["reasons"] == ["expected healthy; observed unhealthy"]
    assert not gate_passes(list(checks.values()))
    assert json.loads((tmp_path / "rollouts.jsonl").read_text()) == record


def test_expected_fault_must_actually_occur(tmp_path, record):
    checks = inspect(tmp_path, [record], {"rollout_ended_on_failed_model_call": "unhealthy"})
    assert checks["rollout_ended_on_failed_model_call"]["reasons"] == ["expected unhealthy; observed healthy"]


@pytest.mark.parametrize(
    "name,status,aggregate_usage",
    [
        ("retry_429", 429, None),
        ("retry_500", 500, None),
        ("model_error", 400, 0),
        ("model_error", 400, None),
        ("model_error", 400, 1),
    ],
)
def test_owned_http_failures_match_the_declared_scenario(tmp_path, record, name, status, aggregate_usage):
    scenario = next(s for s in SCENARIOS if s.name == name)
    trajectory = record["ng_trajectory"]
    failed = deepcopy(trajectory["model_calls"][0])
    failed.update(model_call_id="failed-attempt", started_at=0.0, completed_at=0.5, token_stats={})
    failed["response_metadata"].update(
        response_id=None, status_code=status, error_category="http_error", response_status=None, finish_reason=None
    )
    failed["response"] = {"error": {"message": "injected failure"}}
    ref = {"model_call_id": "failed-attempt", "model_ref": failed["response_metadata"]["model_ref"]}
    if scenario.terminal_error:
        trajectory["model_calls"] = [failed]
        trajectory["turns"] = []
        trajectory["invocations"][0]["model_calls"] = [ref]
        record["response"] = {"output": []}
        if aggregate_usage is not None:
            record["response"]["usage"] = {
                "input_tokens": aggregate_usage,
                "output_tokens": 0,
                "total_tokens": aggregate_usage,
            }
    else:
        trajectory["model_calls"].insert(0, failed)
        trajectory["turns"][0]["model_calls"].insert(0, ref)
        trajectory["invocations"][0]["model_calls"].insert(0, ref)
    checks = inspect(tmp_path, [record], scenario.health_expectations, steps=scenario.steps)
    if scenario.terminal_error and aggregate_usage != 0:
        assert [key for key, value in checks.items() if value["status"] == "fail"] == ["rollout_token_count_mismatch"]
        actual = "unobserved" if aggregate_usage is None else "unhealthy"
        assert checks["rollout_token_count_mismatch"]["reasons"] == [f"expected healthy; observed {actual}"]
    else:
        assert gate_passes(list(checks.values())), checks


@pytest.mark.parametrize("delivery", ["missing", "empty", "malformed", "failure_sidecar"])
def test_missing_or_invalid_evidence_cannot_satisfy_health_expectations(tmp_path, delivery):
    paths = []
    if delivery != "missing":
        path = tmp_path / ("rollouts_failures.jsonl" if delivery == "failure_sidecar" else "rollouts.jsonl")
        path.write_text(
            {"empty": "", "malformed": "not json\n", "failure_sidecar": '{"failure": "error"}\n'}[delivery]
        )
        paths.append(path)
    checks = health.inspect_health(
        paths, output=tmp_path / "health", expectations={"rollout_token_count_mismatch": "unobserved"}
    )
    assert not gate_passes(checks)
    token_check = next(c for c in checks if c["id"] == "health.rollout_token_count_mismatch")
    assert token_check["status"] == ("fail" if delivery in ("missing", "empty") else "pass")


def test_every_record_must_match_not_only_the_overall_unhealthy_verdict(tmp_path, record):
    other = deepcopy(record)
    other["response"]["usage"]["input_tokens"] = 999
    checks = inspect(tmp_path, [record, other], {"rollout_token_count_mismatch": "unhealthy"})
    assert checks["rollout_token_count_mismatch"]["status"] == "fail"
    assert checks["rollout_token_count_mismatch"]["reasons"] == ["expected unhealthy; observed healthy"]


@pytest.mark.parametrize("expectations", [{"removed_check": "unhealthy"}, {"record_unreadable": "anything"}])
def test_invalid_expectations_are_configuration_errors(tmp_path, expectations):
    with pytest.raises(ValueError, match="health expectations"):
        health.inspect_health([], output=tmp_path / "health", expectations=expectations)


def test_registry_addition_is_included_and_a_missing_engine_result_fails(tmp_path, record, monkeypatch):
    spec = CheckSpec(
        id="new_check", evaluation_scope=CheckScope.ROLLOUT, subject=CheckSubject.ROLLOUT, reads=frozenset()
    )
    monkeypatch.setattr(health, "CHECK_REGISTRY", (*health.CHECK_REGISTRY, spec))
    checks = inspect(tmp_path, [record])
    assert checks["new_check"]["status"] == "fail"
    assert "expected healthy; observed no health result" in checks["new_check"]["reasons"][0]


def test_step_scope_is_explicit_and_never_inferred_from_missing_turns(tmp_path, record):
    record["ng_trajectory"]["turns"] = []
    checks = inspect(tmp_path, [record])
    assert checks["rollout_missing_agent_turns"]["status"] == "fail"
    checks = inspect(tmp_path, [record], steps=False)
    assert checks["rollout_missing_agent_turns"]["status"] == "not_applicable"
    assert checks["agent_turn_hollow"]["status"] == "not_applicable"
    assert checks["rollout_token_count_mismatch"]["status"] == "pass"
    record["response"]["usage"]["input_tokens"] = 999
    checks = inspect(tmp_path, [record], steps=False)
    assert checks["rollout_token_count_mismatch"]["reasons"] == ["expected healthy; observed unhealthy"]
