# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Persisted synthetic producer, collector, and gate tests without external services."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from nemo_gym.orchestration.recovery import (
    COLLECT_SCRIPT,
    CollectedEvidence,
    decide_from_collected,
    gate_summary_from_collected,
    response_health,
    trial_health,
)
from nemo_gym.rollout_health import run_health_checks
from nemo_gym.trajectory.trajectory_record import TrajectoryRecord


def write_json(path, value):
    path.write_text(json.dumps(value) + "\n")


def row(repeat=0, explicit=False, task=0):
    suffix = f"{task}-{repeat}"
    rid = f"explicit-{suffix}" if explicit else suffix
    ref = {
        "model_call_id": f"call-{suffix}",
        "response_id": f"response-{suffix}",
        "model_ref": {"type": "responses_api_models", "name": "policy"},
    }
    message = {
        "type": "message",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": "synthetic wrong answer"}],
    }
    meta = {
        "response_id": ref["response_id"],
        "model_ref": ref["model_ref"],
        "status_code": 200,
        "response_status": "completed",
        "error_category": None,
        "failure_report": {"events": {}, "omitted": False, "failure_id": None},
        "model_operations": [
            {
                "operation_id": f"operation-{suffix}",
                "outcome": "completed",
                "final_attempt_id": f"attempt-{suffix}",
                "failure_id": None,
                "http_attempts": 1,
                "transport_failures": 0,
            }
        ],
    }
    trajectory = TrajectoryRecord.model_validate(
        {
            "task_id": str(task),
            "rollout_id": rid,
            "invocations": [
                {
                    "invocation_id": f"agent-{suffix}",
                    "status": "completed",
                    "in_session": False,
                    "verification_skipped": False,
                    "model_calls": [ref],
                }
            ],
            "verifier_invocations": [
                {
                    "invocation_id": f"verifier-{suffix}",
                    "status": "completed",
                    "failure_kind": None,
                }
            ],
            "model_calls": [
                {
                    "model_call_id": ref["model_call_id"],
                    "response_metadata": meta,
                    "request": {"input": [{"role": "user", "content": "synthetic question"}]},
                    "response": {
                        "id": ref["response_id"],
                        "status": "completed",
                        "output": [message],
                    },
                }
            ],
            "turns": [
                {
                    "invocation_id": f"agent-{suffix}",
                    "task_id": str(task),
                    "rollout_id": rid,
                    "turn_no": 1,
                    "timestamp": 1.0,
                    "step_count": 1,
                    "answer": [message],
                    "model_calls": [ref],
                }
            ],
            "tool_calls": [],
            "gaps": [],
        }
    ).model_dump(mode="json")
    result = {
        "_ng_task_index": task,
        "_ng_rollout_index": repeat,
        "reward": 0.0,
        "agent_ref": {"name": "agent"},
        "_ng_environment_server": "environment",
        "_ng_result_type": "legacy_agent",
        "ng_trajectory": trajectory,
    }
    if explicit:
        result["_ng_rollout_id"] = rid
    return result


def save_rows(a, rows):
    (a / "rollouts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


def setup(tmp_path, *, explicit=False, n=2):
    a = tmp_path / "artifacts"
    a.mkdir()
    rows = [row(i, explicit) for i in range(n)]
    save_rows(a, rows)
    inventory = [{k: r[k] for k in ("_ng_task_index", "_ng_rollout_index", "_ng_rollout_id") if k in r} for r in rows]
    (a / "rollouts_materialized_inputs.jsonl").write_text("".join(json.dumps(r) + "\n" for r in inventory))
    (a / "rollouts_failures.jsonl").write_text("")
    write_json(
        a / "rollouts_resolved_config.yaml",
        {
            "agent": {
                "responses_api_agents": {
                    "simple_agent": {
                        "resources_server": {"name": "resource"},
                        "skip_verification": False,
                    }
                }
            },
            "resource": {"resources_servers": {"mcqa": {}}},
            "environment": {"environment_servers": {"legacy_agent": {}}},
        },
    )
    write_json(a / "rollouts_aggregate_metrics.json", {"reward": 0.0})
    produced = run_health_checks(a / "rollouts.jsonl", workers=1)
    assert produced.summary["run"]["stats"]["responses"] == {
        "rollouts": n,
        "unknown": 0,
        "selected": n,
        "faulted": 0,
        "duplicated": 0,
    }
    return a, rows


def collect(a, hook=""):
    # Tracking evidence is local synthetic input, not an actual backend receipt or observation.
    names = {
        "rollouts_jsonl_fpath": "rollouts.jsonl",
        "resolved_config_yaml_fpath": "rollouts_resolved_config.yaml",
        "aggregate_metrics_json_fpath": "rollouts_aggregate_metrics.json",
        "failures_jsonl_fpath": "rollouts_failures.jsonl",
        "materialized_inputs_jsonl_fpath": "rollouts_materialized_inputs.jsonl",
    }
    write_json(
        a / "rollouts_mlflow_export_done",
        {
            "version": 1,
            "run_id": "synthetic-export",
            "uploaded_rollouts": True,
            "rollout_projection_version": 1,
            "artifacts": {k: hashlib.sha256((a / name).read_bytes()).hexdigest() for k, name in names.items()},
        },
    )
    prefix = ""
    # The collector subprocess has the same no-network restriction as this test process.
    prefix += "\nimport builtins\noriginal_import = builtins.__import__\ndef limited(name, *a, **k):\n    if name.startswith('nemo_gym'): raise AssertionError('Gym unavailable on collector host')\n    return original_import(name, *a, **k)\nbuiltins.__import__ = limited\n"
    prefix += "\nimport socket\ndef denied(*a, **k): raise AssertionError('network forbidden')\nsocket.socket.connect = denied\nsocket.create_connection = denied\n"
    proc = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            prefix + hook + "\n" + COLLECT_SCRIPT,
            str(a.parent),
            '["reward"]',
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return CollectedEvidence.model_validate_json(proc.stdout)


def evaluate(c):
    d = decide_from_collected(c, max_infra_retries=1, max_walltime_s=None, live_job=False)
    task = gate_summary_from_collected(
        d,
        c,
        max_unhealthy_fraction=0.02,
        mlflow_status="FINISHED",
        mlflow_run_id="synthetic-export",
    )
    return d, task


def snapshot(a):
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in a.iterdir() if p.is_file()}


def test_legacy_ordinary_zero_keeps_score_but_unobserved_grading_is_unknown(tmp_path):
    a, _ = setup(tmp_path)
    c = collect(a)
    before = snapshot(a)
    d, task = evaluate(c)
    assert d.expected == 2 and d.counts == {"scored": 2}
    assert response_health(d, c.quality)[0] is True and trial_health(d, c)[0] is None
    assert task["status"] == "FAILED" and task["score"] == 0.0
    assert d.action.value == "noop"  # Unknown grading coverage never requests another generation.
    assert snapshot(a) == before


def test_file_changed_after_producer_report_cannot_pass(tmp_path):
    a, rows = setup(tmp_path)
    original_quality = (a / "quality_summary.json").read_bytes()
    rows[0]["ng_trajectory"]["model_calls"][0]["response"]["output"] = []
    save_rows(a, rows)
    before = snapshot(a)
    c = collect(a)
    d, task = evaluate(c)
    print(
        "STALE",
        json.dumps(
            {
                "response": response_health(d, c.quality),
                "trial": trial_health(d, c),
                "task": task,
            },
            sort_keys=True,
        ),
    )
    assert any(error.startswith("response/trial health unknown") for error in task["errors"])
    assert not any(warning.startswith("health gate within") for warning in task["warnings"])
    assert c.quality["source_matches"] is False
    # The stale counters are retained as history, never recomputed or accepted by the remote reader.
    assert c.quality["responses"]["selected"] == 2 and c.quality["responses"]["faulted"] == 0
    assert task["status"] == "FAILED"
    assert (a / "quality_summary.json").read_bytes() == original_quality
    assert all(snapshot(a)[name] == value for name, value in before.items())


def test_valid_explicit_id_survives_collection(tmp_path):
    a, rows = setup(tmp_path, explicit=True)
    c = collect(a)
    d, task = evaluate(c)
    print("EXPLICIT", json.dumps({"trial": trial_health(d, c), "status": task["status"]}))
    assert [r.get("_ng_rollout_id") for r in c.results] == [r["_ng_rollout_id"] for r in rows]
    assert d.expected == 2 and d.counts == {"scored": 2}
    assert trial_health(d, c)[0] is None and task["status"] == "FAILED"


@pytest.mark.parametrize("sources", [None, [], ["other.jsonl"], ["rollouts.jsonl", "rollouts.jsonl"]])
def test_missing_wrong_or_ambiguous_source_remains_nonaccepting(tmp_path, sources):
    a, _ = setup(tmp_path)
    q = json.loads((a / "quality_summary.json").read_text())
    if sources == ["other.jsonl"]:
        (a / "other.jsonl").write_bytes((a / "rollouts.jsonl").read_bytes())
        run_health_checks(a / "other.jsonl", output_dir=a, workers=1)
    elif sources == ["rollouts.jsonl", "rollouts.jsonl"]:
        run_health_checks([a / "rollouts.jsonl", a / "rollouts.jsonl"], output_dir=a, workers=1)
    else:
        # Missing source metadata is an injected malformed persisted-report control.
        if sources is None:
            q["run"].pop("rollouts")
        else:
            q["run"]["rollouts"] = sources
        write_json(a / "quality_summary.json", q)
    c = collect(a)
    d, task = evaluate(c)
    assert c.quality["source_matches"] is False
    assert task["status"] == "FAILED"
    assert any("not bound" in e or "unknown" in e for e in task["errors"])


def test_missing_report_remains_nonaccepting(tmp_path):
    a, _ = setup(tmp_path)
    (a / "quality_summary.json").unlink()
    c = collect(a)
    _, task = evaluate(c)
    assert c.quality is None and task["status"] == "FAILED"


def test_saved_answer_and_spent_budget_files_unchanged(tmp_path):
    a, _ = setup(tmp_path)
    failed = {
        "_ng_task_index": 2,
        "_ng_rollout_index": 0,
        "_ng_failure_class": "judge_failed",
        "failure_kind": "judge_failed",
        "response": {
            "id": "saved",
            "object": "response",
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "synthetic saved answer must not regenerate"}],
                }
            ],
        },
    }
    (a / "rollouts_failures.jsonl").write_text(json.dumps(failed) + "\n")
    for name, text in {
        ".gym_accumulated_walltime": "3607",
        ".gym_infra_retries": "1",
        ".gym_job_chain": "synthetic-job",
    }.items():
        (a.parent / name).write_text(text)
    protected = [
        a / "rollouts.jsonl",
        a / "rollouts_materialized_inputs.jsonl",
        a / "rollouts_failures.jsonl",
        a.parent / ".gym_accumulated_walltime",
        a.parent / ".gym_infra_retries",
        a.parent / ".gym_job_chain",
    ]
    before = {str(p): p.read_bytes() for p in protected}
    c = collect(a)
    from nemo_gym.rollout_recovery import has_saved_generation

    assert has_saved_generation(c.failures[0])
    assert "synthetic saved answer must not regenerate" not in json.dumps(c.failures[0])
    assert {str(p): p.read_bytes() for p in protected} == before


def test_change_during_collection_remains_nonaccepting(tmp_path):
    a, _ = setup(tmp_path)
    hook = """
import json, os, sys
original_loads = json.loads
changed = False
def changing(value, *args, **kwargs):
    global changed
    result = original_loads(value, *args, **kwargs)
    if isinstance(value, bytes) and isinstance(result, dict) and "_ng_task_index" in result and not changed:
        changed = True
        with open(os.path.join(sys.argv[1], "artifacts", "rollouts.jsonl"), "a") as stream:
            stream.write("\\n")
    return result
json.loads = changing
"""
    c = collect(a, hook)
    _, task = evaluate(c)
    assert c.quality["source_matches"] is False and task["status"] == "FAILED"


@pytest.mark.parametrize("binding", [None, [], "malformed", [{}], [{"rollouts_jsonl_fpath": True}]])
def test_missing_or_malformed_binding_remains_nonaccepting(tmp_path, binding):
    a, _ = setup(tmp_path)
    q = json.loads((a / "quality_summary.json").read_text())
    q["run"]["provenance"]["rollout_artifact_digests"] = binding
    write_json(a / "quality_summary.json", q)
    c = collect(a)
    _, task = evaluate(c)
    assert c.quality["source_matches"] is False and task["status"] == "FAILED"


@pytest.mark.parametrize(
    "name",
    [
        "rollouts.jsonl",
        "rollouts_failures.jsonl",
        "rollouts_materialized_inputs.jsonl",
        "rollouts_resolved_config.yaml",
    ],
)
def test_any_gate_input_changed_after_report_is_unbound(tmp_path, name):
    a, _ = setup(tmp_path)
    path = a / name
    path.write_bytes(path.read_bytes() + b"\n")
    before = snapshot(a)
    c = collect(a)
    _, task = evaluate(c)
    assert c.quality["source_matches"] is False and task["status"] == "FAILED"
    assert all(snapshot(a)[name] == value for name, value in before.items())


@pytest.mark.parametrize(
    "name",
    [
        "rollouts.jsonl",
        "rollouts_failures.jsonl",
        "rollouts_materialized_inputs.jsonl",
        "rollouts_resolved_config.yaml",
    ],
)
def test_inputs_changed_while_producing_report_are_not_bound(tmp_path, monkeypatch, name):
    import nemo_gym.rollout_health as producer

    a, _ = setup(tmp_path)
    original = producer._index_jsonl

    def changing(paths):
        result = original(paths)
        path = a / name
        path.write_bytes(path.read_bytes() + b"\n")
        return result

    monkeypatch.setattr(producer, "_index_jsonl", changing)
    result = run_health_checks(a / "rollouts.jsonl", workers=1)
    assert "rollout_artifact_digests" not in result.summary["run"]["provenance"]
    c = collect(a)
    _, task = evaluate(c)
    assert c.quality["source_matches"] is False and task["status"] == "FAILED"


@pytest.mark.parametrize("n,accept", [(49, False), (50, False), (51, True)])
def test_persisted_response_denominator_and_strict_boundary(tmp_path, n, accept):
    a, rows = setup(tmp_path, n=n)
    rows[0]["ng_trajectory"]["model_calls"][0]["response"]["output"] = []
    save_rows(a, rows)
    run_health_checks(a / "rollouts.jsonl", workers=1)
    before = (a / "rollouts.jsonl").read_bytes()
    c = collect(a)
    d, task = evaluate(c)
    print(
        "BOUNDARY",
        json.dumps(
            {
                "expected_trials": d.expected,
                "responses": c.quality["responses"],
                "status": task["status"],
            }
        ),
    )
    assert d.expected == n and d.counts == {"scored": n} and d.action.value == "noop"
    assert c.quality["responses"]["selected"] == n and c.quality["responses"]["faulted"] == 1
    assert response_health(d, c.quality)[0] is accept
    # This historical fixture has no grading operation. Response threshold is still exact above;
    # absence of grading evidence now leaves the full gate nonaccepting even below that threshold.
    assert trial_health(d, c)[0] is None
    assert task["status"] == "FAILED"
    assert (a / "rollouts.jsonl").read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    ["missing-key", "extra-key", "bool-hash", "bad-hash", "wrong-hash", "null-rollouts", "duplicate-binding"],
)
def test_binding_requires_exact_keys_types_and_current_digests(tmp_path, mutation):
    a, _ = setup(tmp_path)
    path = a / "quality_summary.json"
    summary = json.loads(path.read_text())
    bindings = summary["run"]["provenance"]["rollout_artifact_digests"]
    binding = bindings[0]
    if mutation == "missing-key":
        binding.pop("failures_jsonl_fpath")
    elif mutation == "extra-key":
        binding["unexpected"] = None
    elif mutation == "duplicate-binding":
        bindings.append(dict(binding))
    else:
        binding["rollouts_jsonl_fpath"] = {
            "bool-hash": True,
            "bad-hash": "z" * 64,
            "wrong-hash": "0" * 64,
            "null-rollouts": None,
        }[mutation]
    write_json(path, summary)
    c = collect(a)
    _, task = evaluate(c)
    assert c.quality["source_matches"] is False and task["status"] == "FAILED"


@pytest.mark.parametrize(
    "broken",
    [
        {"run": []},
        {"run": {"artifacts": "invalid"}},
        {"run": {"stats": [1]}},
        {"run": {"stats": {"model_call_errors": "invalid"}}},
    ],
)
def test_malformed_optional_quality_does_not_erase_saved_score(tmp_path, broken):
    a, _ = setup(tmp_path)
    write_json(a / "quality_summary.json", broken)
    c = collect(a)
    before = snapshot(a)
    d, task = evaluate(c)
    assert c.quality is None
    assert d.action.value == "noop" and d.counts == {"scored": 2}
    assert task["status"] == "FAILED" and task["score"] == 0.0
    assert snapshot(a) == before


@pytest.mark.parametrize(
    "identity_fields,task_id",
    [
        ({"task_id": "dataset-task"}, "dataset-task"),
        ({"problem_id": "dataset-problem"}, "dataset-problem"),
        ({"instance_id": "dataset-instance"}, "dataset-instance"),
        ({"task_id": 17}, "17"),
        ({"task_id": "first", "problem_id": "second", "instance_id": "third"}, "first"),
        ({"trajectory_identity": {"task_id": "bound-task", "rollout_id": "0-0"}, "task_id": "fallback"}, "bound-task"),
    ],
)
@pytest.mark.parametrize("tamper", [None, "trajectory", "turn"])
def test_legacy_task_identity_is_bound_to_saved_input(tmp_path, identity_fields, task_id, tamper):
    a, rows = setup(tmp_path, n=1)
    inputs = [json.loads(line) for line in (a / "rollouts_materialized_inputs.jsonl").read_text().splitlines()]
    inputs[0].update(identity_fields)
    (a / "rollouts_materialized_inputs.jsonl").write_text(json.dumps(inputs[0]) + "\n")
    rows[0]["ng_trajectory"]["task_id"] = task_id
    rows[0]["ng_trajectory"]["turns"][0]["task_id"] = task_id
    if tamper == "trajectory":
        rows[0]["ng_trajectory"]["task_id"] = "unrelated"
    if tamper == "turn":
        rows[0]["ng_trajectory"]["turns"][0]["task_id"] = "unrelated"
    save_rows(a, rows)
    run_health_checks(a / "rollouts.jsonl", workers=1)
    c = collect(a)
    before = snapshot(a)
    d, task = evaluate(c)
    assert d.action.value == "noop" and d.counts == {"scored": 1}
    assert task["score"] == 0.0
    assert c.materialized[0]["_ng_collected_task_id"] == task_id
    # Legacy grading absence remains unknown independently of identity; current bound producer
    # identity mismatches have positive/negative controls in test_tool_trial_coverage.
    assert trial_health(d, c)[0] is None and task["status"] == "FAILED"
    assert snapshot(a) == before


@pytest.mark.parametrize(
    "counters,expected",
    [
        ({}, None),
        ({"recovered_attempt_errors": "1", "recovered_operations": 0}, None),
        ({"recovered_attempt_errors": 0, "recovered_operations": True}, None),
        ({"recovered_attempt_errors": -1, "recovered_operations": 0}, None),
        (False, None),
        ({"recovered_attempt_errors": 0, "recovered_operations": 0}, True),
        ({"recovered_attempt_errors": 2, "recovered_operations": 1}, None),
    ],
)
def test_recovery_counters_preserve_observed_mcqa_gate(counters, expected):
    collected = CollectedEvidence.model_validate_json(
        (Path(__file__).parent / "fixtures/cpu-grading-mcqa-healthy.json").read_text()
    )
    decision = decide_from_collected(collected, max_infra_retries=1, max_walltime_s=None, live_job=False)
    assert trial_health(decision, collected)[0] is True
    collected.quality["observed_upstream_recovery"] = counters
    assert trial_health(decision, collected)[0] is expected
    assert collected.quality["observed_upstream_recovery"] == counters
    assert decision.action.value == "noop"
