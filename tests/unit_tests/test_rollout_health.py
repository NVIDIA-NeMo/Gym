# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from asyncio import Future
from copy import deepcopy
from pathlib import Path
from threading import get_ident

import orjson
import pytest

import nemo_gym.cli.eval as cli_eval
import nemo_gym.cli.main as cli_main
import nemo_gym.health.checks as health_checks
import nemo_gym.rollout_collection as rollout_collection
import nemo_gym.rollout_health as health
from nemo_gym.observability.model_calls import build_model_call_record
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper, _CompletedRollout
from nemo_gym.rollout_health import CHECK_REGISTRY, run_health_checks
from nemo_gym.trajectory import TrajectoryRecord


MODEL_CALL_CHECKS = {
    "model_call_zero_completion_tokens",
    "model_call_missing_token_counts",
    "trajectory_capture_mismatch",
    "rollout_token_count_mismatch",
    "model_call_runaway_generation",
}
RUNNER_CHECKS = {"check_execution_error", "record_unreadable"}
# Unlike the bound-call checks, this one reads the captured calls directly, so it
# is unobserved only when no canonical call exists -- not when none of them bind.
NO_CANONICAL_CALL_CHECKS = MODEL_CALL_CHECKS | {"rollout_ended_on_failed_model_call"}


def _record(
    task: int,
    rollout: int,
    *,
    answer: str | None = "ok",
    refs: list[dict] | None = None,
    include_turn: bool = True,
    include_response_output: bool = True,
    usage: dict | None = None,
) -> dict:
    model_refs = refs if refs is not None else [{"model_call_id": "c1"}]
    trajectory = {
        "task_id": str(task),
        "rollout_id": f"{task}-{rollout}",
        "turns": [],
        "model_calls": [],
    }
    if include_turn:
        trajectory["turns"] = [
            {
                "invocation_id": "root",
                "task_id": str(task),
                "rollout_id": f"{task}-{rollout}",
                "turn_no": 1,
                "timestamp": 1.0,
                "answer": answer,
                "step_count": 1,
                "model_calls": model_refs,
            }
        ]
    response = {
        "output": (
            [{"type": "message", "role": "assistant", "content": answer or ""}] if include_response_output else []
        )
    }
    if usage is not None:
        response["usage"] = usage
    return {
        "_ng_task_index": task,
        "_ng_rollout_index": rollout,
        "response": response,
        "ng_trajectory": trajectory,
    }


def _call(**updates) -> dict:
    call = {
        "call_index": 0,
        "model_call_id": "c1",
        "response_id": "r1",
        "status_code": 200,
        "response_status": "completed",
        "finish_reason": "stop",
        "tokens_in": 3,
        "tokens_out": 2,
        "request": {"input": "question"},
        "response": {"output_text": "ok"},
    }
    call.update(updates)
    return call


def _trajectory_call(call: dict) -> dict:
    return {
        "model_call_id": call.get("model_call_id"),
        "client_session_id": call.get("client_session_id"),
        "started_at": call.get("started_at"),
        "request": call.get("request"),
        "response": call.get("response"),
        "response_metadata": {
            "response_id": call.get("response_id"),
            "model_ref": call.get("model_ref"),
            "status_code": call.get("status_code"),
            "response_status": call.get("response_status"),
            "finish_reason": call.get("finish_reason"),
            "error_category": call.get("error_category"),
            "upstream_attempted": call.get("upstream_attempted"),
            "upstream_status_code": call.get("upstream_status_code"),
            "local_response_reason": call.get("local_response_reason"),
        },
        "token_stats": {
            "prompt_tokens": call.get("tokens_in"),
            "completion_tokens": call.get("tokens_out"),
        },
    }


def _write_fixture(root: Path, rows: list[tuple[dict, list[dict]]]) -> Path:
    rollout_path = root / "rollouts.jsonl"
    with rollout_path.open("wb") as rollouts:
        for record, calls in rows:
            stored = deepcopy(record)
            stored["ng_trajectory"]["model_calls"] = [_trajectory_call(call) for call in calls]
            rollouts.write(orjson.dumps(stored, option=orjson.OPT_APPEND_NEWLINE))
    return rollout_path


def test_check_ids_encode_subject_without_replacing_evaluation_scope() -> None:
    assert all(spec.id.startswith(f"{spec.subject.value}_") for spec in CHECK_REGISTRY)
    by_id = {spec.id: spec for spec in CHECK_REGISTRY}
    assert by_id["rollout_ended_on_failed_model_call"].evaluation_scope == health.CheckScope.ROLLOUT
    assert by_id["rollout_ended_on_failed_model_call"].subject == health.CheckSubject.ROLLOUT
    assert by_id["task_consistently_unhealthy"].evaluation_scope == health.CheckScope.TASK
    assert by_id["task_consistently_unhealthy"].subject == health.CheckSubject.TASK
    assert by_id["trajectory_capture_mismatch"].reads == frozenset(
        {health.CheckInput.RECORD, health.CheckInput.TRAJECTORY, health.CheckInput.OWNED_MODEL_CALLS}
    )
    assert by_id["agent_turn_hollow"].reads == frozenset(
        {health.CheckInput.RECORD, health.CheckInput.TRAJECTORY, health.CheckInput.AGENT_TURNS}
    )
    assert by_id["rollout_token_count_mismatch"].reads == frozenset(
        {health.CheckInput.RECORD, health.CheckInput.TRAJECTORY, health.CheckInput.OWNED_MODEL_CALLS}
    )
    with pytest.raises(ValueError, match="cannot read both"):
        health.CheckSpec(
            id="model_call_invalid_binding_inputs",
            evaluation_scope=health.CheckScope.ROLLOUT,
            subject=health.CheckSubject.MODEL_CALL,
            reads=frozenset({health.CheckInput.BOUND_CALLS, health.CheckInput.OWNED_MODEL_CALLS}),
        )


def test_invocation_owned_calls_are_observed_without_turns(tmp_path: Path) -> None:
    record = _record(0, 0, include_turn=False)
    record["ng_trajectory"]["invocations"] = [
        {"kind": "agent_invocation", "invocation_id": "root", "model_calls": [{"model_call_id": "c1"}]}
    ]
    record["ng_trajectory"]["gaps"] = [{"code": "turns_unavailable"}]
    rollout_path = _write_fixture(tmp_path, [(record, [_call()])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    invocation_checks = {
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "trajectory_capture_mismatch",
        "model_call_runaway_generation",
    }
    assert not invocation_checks & set(digest.unobserved)
    assert set(digest.unobserved) == {
        "rollout_missing_agent_turns",
        "agent_turn_hollow",
        "rollout_token_count_mismatch",
    }
    assert digest.verdict == "unobserved"


@pytest.mark.parametrize(
    "check,refs,call",
    [
        ("model_call_zero_completion_tokens", [{"model_call_id": "c1"}], _call(tokens_out=0)),
        ("model_call_missing_token_counts", [{"model_call_id": "c1"}], _call(tokens_out=None)),
        ("trajectory_capture_mismatch", [{"model_call_id": "missing"}], _call()),
        ("model_call_runaway_generation", [{"model_call_id": "c1"}], _call(finish_reason="length", response={})),
    ],
)
def test_invocation_owned_call_checks_fire_without_turns(
    tmp_path: Path, check: str, refs: list[dict], call: dict
) -> None:
    record = _record(0, 0, include_turn=False)
    record["ng_trajectory"]["invocations"] = [
        {"kind": "agent_invocation", "invocation_id": "root", "model_calls": refs}
    ]
    record["ng_trajectory"]["gaps"] = [{"code": "turns_unavailable"}]
    rollout_path = _write_fixture(tmp_path, [(record, [call])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert check in {finding.check for finding in digest.findings}
    assert digest.verdict == "unhealthy"


def test_turn_and_invocation_reference_evaluate_one_call(tmp_path: Path) -> None:
    model_ref = {"type": "responses_api_models", "name": "policy_model"}
    record = _record(0, 0, refs=[{"model_ref": model_ref, "response_id": "r1"}])
    record["ng_trajectory"]["invocations"] = [
        {"kind": "agent_invocation", "invocation_id": "root", "model_calls": [{"model_call_id": "c1"}]}
    ]
    rollout_path = _write_fixture(
        tmp_path,
        [(record, [_call(model_ref=model_ref, tokens_out=0, status_code=500)])],
    )

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert [finding.check for finding in digest.findings].count("model_call_zero_completion_tokens") == 1


def test_unbindable_turn_reference_does_not_shadow_invocation_reference(tmp_path: Path) -> None:
    record = _record(0, 0, refs=[{"model_call_id": "c1", "response_id": "stale-response"}])
    record["ng_trajectory"]["invocations"] = [
        {"kind": "agent_invocation", "invocation_id": "root", "model_calls": [{"model_call_id": "c1"}]}
    ]
    rollout_path = _write_fixture(tmp_path, [(record, [_call(tokens_out=0)])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert [finding.check for finding in digest.findings].count("trajectory_capture_mismatch") == 1
    failed = [finding for finding in digest.findings if finding.check == "model_call_zero_completion_tokens"]
    assert len(failed) == 1
    assert failed[0].locator == {"call_id": "c1"}


def test_task_no_successful_model_calls_remains_turn_bound(tmp_path: Path) -> None:
    rows = []
    for rollout_index in range(2):
        record = _record(8, rollout_index, usage={"input_tokens": 3, "output_tokens": 2})
        record["ng_trajectory"]["invocations"] = [
            {
                "kind": "agent_invocation",
                "invocation_id": "root",
                "model_calls": [{"model_call_id": "c1"}, {"model_call_id": "support"}],
            }
        ]
        rows.append(
            (
                record,
                [
                    _call(model_call_id="c1", status_code=500),
                    _call(model_call_id="support", response_id="support-response"),
                ],
            )
        )
    result = run_health_checks(_write_fixture(tmp_path, rows), workers=1)

    assert all(digest.successful_model_calls == 0 for digest in result.rollouts)
    assert "task_no_successful_model_calls" in result.summary["tasks"]["8"]["flags"]


def test_all_registered_semantic_checks_fire_on_synthetic_artifacts(tmp_path: Path) -> None:
    rows = [
        (_record(0, 0, include_turn=False, include_response_output=False), [_call()]),
        (_record(0, 1, include_turn=False, include_response_output=False), [_call()]),
        (_record(1, 0, answer=None), [_call()]),
        (
            _record(2, 0, usage={"input_tokens": 3, "output_tokens": 0}),
            [_call(tokens_out=0, response={"output_text": ""})],
        ),
        (_record(3, 0, usage=None), [_call(tokens_out=None)]),
        (_record(4, 0, refs=[{"model_call_id": "missing"}]), [_call()]),
        (
            _record(5, 0, usage={"input_tokens": 3, "output_tokens": 2}),
            [_call(status_code=500, error_category="upstream")],
        ),
        (_record(6, 0, usage={"input_tokens": 99, "output_tokens": 99}), [_call()]),
        (
            _record(7, 0, usage={"input_tokens": 3, "output_tokens": 2}),
            [_call(finish_reason="length", response={})],
        ),
        (_record(8, 0, usage={"input_tokens": 3, "output_tokens": 2}), [_call(status_code=500)]),
        (_record(8, 1, usage={"input_tokens": 3, "output_tokens": 2}), [_call(status_code=408)]),
    ]
    rows.append(deepcopy(rows[0]))
    rollout_path = _write_fixture(tmp_path, rows)

    result = run_health_checks(rollout_path, workers=2)

    assert set(result.summary["run"]["issues"]) == {spec.id for spec in CHECK_REGISTRY}
    # Run-scope checks judge the run accounting, not records; TestRunExcessiveFailuresCheck covers them.
    record_checks = [spec for spec in CHECK_REGISTRY if spec.evaluation_scope != health.CheckScope.RUN]
    assert all(result.summary["run"]["issues"][spec.id] > 0 for spec in record_checks if spec.id not in RUNNER_CHECKS)
    assert all(result.summary["run"]["issues"][check_id] == 0 for check_id in RUNNER_CHECKS)
    assert result.summary["tasks"]["0"]["flags"] == ["task_consistently_unhealthy"]
    assert "task_no_successful_model_calls" in result.summary["tasks"]["8"]["flags"]
    assert result.summary_path == tmp_path / "quality_summary.json"
    assert result.verdicts_path == tmp_path / "rollout_verdicts.jsonl"

    summary = json.loads(result.summary_path.read_text())
    assert set(summary) == {"run", "tasks"}
    verdict_rows = [json.loads(line) for line in result.verdicts_path.read_text().splitlines()]
    assert [(row["_ng_task_index"], row["_ng_rollout_index"]) for row in verdict_rows] == sorted(
        (record["_ng_task_index"], record["_ng_rollout_index"]) for record, _ in rows
    )
    assert set(verdict_rows[0]) == {
        "_ng_task_index",
        "_ng_rollout_index",
        "rollout_id",
        "verdict",
        "findings",
        "unobserved",
    }


@pytest.mark.parametrize("state", ["missing model calls", "missing bindings"])
def test_each_canonical_model_call_unobserved_state_is_not_unhealthy(tmp_path: Path, state: str) -> None:
    record = _record(0, 0)
    if state == "missing model calls":
        record["ng_trajectory"]["turns"][0]["model_calls"] = []
    else:
        record["ng_trajectory"]["turns"][0]["model_calls"] = []
        record["ng_trajectory"]["model_calls"] = [_trajectory_call(_call())]
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    result = run_health_checks(rollout_path, workers=1)

    [digest] = result.rollouts
    assert digest.verdict == "unobserved"
    expected = MODEL_CALL_CHECKS if state == "missing bindings" else NO_CANONICAL_CALL_CHECKS
    assert set(digest.unobserved) == expected
    assert not digest.findings
    assert result.summary["run"]["verdicts"] == {"healthy": 0, "unhealthy": 0, "unobserved": 1}


def test_missing_canonical_trajectory_makes_trajectory_checks_unobserved(tmp_path: Path) -> None:
    record = _record(0, 0)
    record.pop("ng_trajectory")
    record["ng_model_call_capture"] = {"calls": [_call()]}
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert digest.verdict == "unobserved"
    assert set(digest.unobserved) == {spec.id for spec in CHECK_REGISTRY if health.CheckInput.TRAJECTORY in spec.reads}
    assert not digest.capture_observed
    assert digest.model_calls == 0
    assert not digest.findings


async def test_health_on_and_off_leave_collection_and_metrics_byte_identical(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setattr(rollout_collection, "get_global_config_dict", lambda: {})
    source = {
        "responses_create_params": {"input": []},
        "agent_ref": {"name": "synthetic-agent"},
    }

    class GoldenHelper(RolloutCollectionHelper):
        def _run_examples_with_metadata(self, examples, *args, **kwargs):
            futures = []
            for example in examples:
                future = Future()
                future.set_result(
                    _CompletedRollout(
                        row=example,
                        result={
                            "response": {
                                "output": [
                                    {
                                        "type": "message",
                                        "role": "assistant",
                                        "content": [{"type": "output_text", "text": "ok"}],
                                    }
                                ],
                                "usage": {"input_tokens": 3, "output_tokens": 1},
                            },
                            "reward": 1.0,
                        },
                        rollout_latency_ms=None,
                    )
                )
                futures.append(future)
            return futures

        async def _call_aggregate_metrics(self, results, rows, output_fpath, *, raise_on_error: bool = True):
            metrics_path = output_fpath.with_stem(output_fpath.stem + "_aggregate_metrics").with_suffix(".json")
            metrics_path.write_bytes(orjson.dumps([{"key_metrics": {"reward": 1.0}}]))
            return metrics_path

    artifacts: dict[bool, dict[str, bytes]] = {}
    for disabled in (False, True):
        run_dir = tmp_path / ("off" if disabled else "on")
        run_dir.mkdir()
        input_path = run_dir / "input.jsonl"
        input_path.write_bytes(orjson.dumps(source, option=orjson.OPT_APPEND_NEWLINE))
        output_path = run_dir / "rollouts.jsonl"
        config = RolloutCollectionConfig(
            input_jsonl_fpath=str(input_path),
            output_jsonl_fpath=str(output_path),
            upload_rollouts=False,
            disable_health_check=disabled,
        )

        await GoldenHelper().run_from_config(config)
        stdout = capsys.readouterr().out

        artifacts[disabled] = {
            "materialized": config.materialized_jsonl_fpath.read_bytes(),
            "rollouts": output_path.read_bytes(),
            "failures": output_path.with_name("rollouts_failures.jsonl").read_bytes(),
            "metrics": output_path.with_name("rollouts_aggregate_metrics.json").read_bytes(),
        }
        assert (run_dir / "quality_summary.json").exists() is not disabled
        assert (run_dir / "rollout_verdicts.jsonl").exists() is not disabled
        if not disabled:
            assert stdout.rstrip().endswith(str(run_dir / "quality_summary.json"))
            assert stdout.index("Finished rollout collection") < stdout.index("Rollout health")

    assert artifacts[False] == artifacts[True]

    failed_run_dir = tmp_path / "health-failure"
    failed_run_dir.mkdir()
    failed_input_path = failed_run_dir / "input.jsonl"
    failed_input_path.write_bytes(orjson.dumps(source, option=orjson.OPT_APPEND_NEWLINE))
    failed_output_path = failed_run_dir / "rollouts.jsonl"
    failed_config = RolloutCollectionConfig(
        input_jsonl_fpath=str(failed_input_path),
        output_jsonl_fpath=str(failed_output_path),
        upload_rollouts=False,
    )

    caller_thread = get_ident()
    health_thread = None

    def broken_health_check(*args, **kwargs):
        nonlocal health_thread
        health_thread = get_ident()
        raise RuntimeError("health failed")

    monkeypatch.setattr(health, "run_health_checks", broken_health_check)
    await GoldenHelper().run_from_config(failed_config)

    assert failed_output_path.exists()
    assert health_thread is not None
    assert health_thread != caller_thread
    assert "Rollout health checks failed after collection" in caplog.text


def test_health_check_cli_accepts_run_dir_workers_and_ignored_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received = {}

    def fake_health_check(
        run_dir,
        *,
        rollout_file=None,
        workers=None,
        ignored_checks=(),
        limits=None,
        json_output=False,
        model_call_capture_dir=None,
    ):
        received.update(
            run_dir=run_dir,
            rollout_file=rollout_file,
            workers=workers,
            ignored_checks=ignored_checks,
            json_output=json_output,
            model_call_capture_dir=model_call_capture_dir,
        )

    monkeypatch.setattr(cli_eval, "health_check_rollouts", fake_health_check)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gym",
            "eval",
            "health-check",
            str(tmp_path),
            "--rollouts-file",
            "evaluator_rollouts.jsonl",
            "--workers",
            "3",
            "--ignore-checks",
            "model_call_missing_token_counts,model_call_zero_completion_tokens",
            "--json",
        ],
    )

    cli_main.main()

    assert received == {
        "run_dir": str(tmp_path),
        "rollout_file": Path("evaluator_rollouts.jsonl"),
        "workers": 3,
        "ignored_checks": ["model_call_missing_token_counts", "model_call_zero_completion_tokens"],
        "json_output": True,
        "model_call_capture_dir": None,
    }


def test_health_check_json_output_is_machine_readable(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(_record(0, 0), option=orjson.OPT_APPEND_NEWLINE))

    result = health.health_check_run_dir(tmp_path, workers=1, json_output=True)

    assert json.loads(capsys.readouterr().out) == result.summary


def test_invalid_canonical_trajectory_is_unreadable_without_fallback(tmp_path: Path) -> None:
    record = _record("task-a", "repeat-a")
    record["ng_trajectory"]["turns"] = ["malformed-turn"]
    record["ng_model_call_capture"] = {"calls": [_call()]}
    record["response"]["output"] = [{"role": "assistant", "content": "fallback answer"}]
    rollout_path = tmp_path / "stored.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    result = run_health_checks(rollout_path, workers=1)

    [digest] = result.rollouts
    assert digest.verdict == "unhealthy"
    assert not digest.capture_observed
    assert digest.model_calls == 0
    assert [finding.check for finding in digest.findings] == ["record_unreadable"]
    assert digest.findings[0].detail["reason"] == "canonical trajectory is unreadable"
    assert set(digest.unobserved) == {spec.id for spec in CHECK_REGISTRY if health.CheckInput.TRAJECTORY in spec.reads}
    assert health_checks._nonempty(123) is False
    assert health_checks._call_ref_key({"response_id": "unqualified-response"}) is None
    assert health_checks._item_has_tool_call("bad") is False


def test_current_trajectory_turn_shape_recognizes_reasoning_and_function_calls(tmp_path: Path) -> None:
    trajectory = TrajectoryRecord.model_validate(
        {
            "task_id": "0",
            "rollout_id": "0-0",
            "turns": [
                {
                    "invocation_id": "root",
                    "task_id": "0",
                    "rollout_id": "0-0",
                    "turn_no": 1,
                    "timestamp": 1.0,
                    "answer": [
                        {
                            "type": "function_call",
                            "call_id": "tool-1",
                            "name": "tool",
                            "arguments": "{}",
                        }
                    ],
                    "reasoning_content": [
                        {
                            "type": "reasoning",
                            "id": "reasoning-1",
                            "summary": [{"type": "summary_text", "text": "thinking"}],
                        }
                    ],
                    "step_count": 1,
                    "model_calls": [{"model_call_id": "call-1"}],
                }
            ],
        }
    )
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "response": {"output": []},
        "ng_trajectory": trajectory.model_dump(mode="json"),
    }
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    result = run_health_checks(rollout_path, workers=1)

    steps = health_checks._agent_steps(record["ng_trajectory"])
    assert len(steps) == 1
    assert steps[0].has_message and steps[0].has_tool_calls
    assert not any(finding.check == "agent_turn_hollow" for finding in result.rollouts[0].findings)


@pytest.mark.parametrize("item_type", sorted(health_checks._AGENT_TOOL_CALL_TYPES))
def test_canonical_turn_tool_call_types_count_as_agent_activity(item_type: str) -> None:
    assert health_checks._item_has_tool_call({"type": item_type})


def test_canonical_turn_refusal_counts_as_message_content() -> None:
    assert health_checks._nonempty({"type": "refusal", "refusal": "cannot comply"})


def test_trajectory_projection_failure_makes_trajectory_checks_unobserved(tmp_path: Path) -> None:
    record = _record(0, 0)
    record["ng_trajectory"]["turns"] = []
    record["ng_trajectory"]["gaps"] = [{"code": "trajectory_projection_failed", "detail": "ValidationError"}]
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert digest.verdict == "unobserved"
    assert set(digest.unobserved) == {spec.id for spec in CHECK_REGISTRY if health.CheckInput.TRAJECTORY in spec.reads}
    assert not digest.findings


def test_missing_canonical_turn_evidence_is_unobserved_not_missing(tmp_path: Path) -> None:
    record = _record(0, 0)
    record["ng_trajectory"]["turns"] = []
    record["ng_trajectory"]["gaps"] = [{"code": "turns_unavailable"}]
    rollout_path = _write_fixture(tmp_path, [(record, [])])

    verdict = run_health_checks(rollout_path, workers=1).rollouts[0]

    assert "rollout_missing_agent_turns" in verdict.unobserved
    assert "agent_turn_hollow" in verdict.unobserved
    assert not {"rollout_missing_agent_turns", "agent_turn_hollow"} & {item.check for item in verdict.findings}


def test_incomplete_canonical_model_call_evidence_is_unobserved_not_mismatched(tmp_path: Path) -> None:
    record = _record(0, 0)
    record["ng_trajectory"]["gaps"] = [{"code": "model_calls_unavailable"}]
    rollout_path = _write_fixture(tmp_path, [(record, [])])

    verdict = run_health_checks(rollout_path, workers=1).rollouts[0]

    assert "trajectory_capture_mismatch" in verdict.unobserved
    assert "trajectory_capture_mismatch" not in {item.check for item in verdict.findings}


def test_canonical_reference_contradiction_gap_is_a_finding(tmp_path: Path) -> None:
    record = _record(0, 0)
    record["ng_trajectory"]["turns"][0]["model_calls"] = []
    record["ng_trajectory"]["gaps"] = [
        {
            "code": "model_call_reference_conflict",
            "invocation_id": "inv-0",
            "detail": "call-conflict",
        }
    ]
    rollout_path = _write_fixture(tmp_path, [(record, [_call(model_call_id="call-other")])])

    verdict = run_health_checks(rollout_path, workers=1).rollouts[0]

    finding = next(item for item in verdict.findings if item.check == "trajectory_capture_mismatch")
    assert finding.locator == {"call_id": "call-conflict"}
    assert finding.detail["kind"] == "conflicting_call_ownership"
    assert finding.detail["observation_gap"] == "model_call_reference_conflict"


def test_ignored_check_is_excluded_from_execution_and_verdict(tmp_path: Path) -> None:
    rollout_path = _write_fixture(
        tmp_path,
        [(_record(0, 0, usage={"input_tokens": 3, "output_tokens": 0}), [_call(tokens_out=0)])],
    )

    result = run_health_checks(
        rollout_path,
        ignored_checks=["model_call_zero_completion_tokens"],
        workers=1,
    )

    [digest] = result.rollouts
    assert digest.verdict == "healthy"
    assert digest.unobserved == []
    assert not any(finding.check == "model_call_zero_completion_tokens" for finding in digest.findings)
    assert result.summary["run"]["ignored_checks"] == ["model_call_zero_completion_tokens"]
    assert result.summary["run"]["artifacts"]["coverage"]["model_call_zero_completion_tokens"] == {
        "evaluated": 0,
        "unobserved": 0,
        "ignored": 1,
    }
    assert "(ignored: model_call_zero_completion_tokens)" in health.format_health_report(result)


def test_ignored_failing_and_task_checks_do_not_emit_findings(tmp_path: Path) -> None:
    rollout_path = _write_fixture(
        tmp_path,
        [
            (_record(0, 0, usage={"input_tokens": 3, "output_tokens": 0}), [_call(tokens_out=0)]),
            (_record(0, 1, usage={"input_tokens": 3, "output_tokens": 0}), [_call(tokens_out=0)]),
        ],
    )

    result = run_health_checks(
        rollout_path,
        ignored_checks=["model_call_zero_completion_tokens", "task_consistently_unhealthy"],
        workers=1,
    )

    assert result.summary["run"]["verdicts"] == {"healthy": 2, "unhealthy": 0, "unobserved": 0}
    assert result.summary["run"]["issues"]["model_call_zero_completion_tokens"] == 0
    assert result.summary["tasks"]["0"]["flags"] == []
    assert result.summary["run"]["artifacts"]["coverage"]["task_consistently_unhealthy"] == {
        "evaluated": 0,
        "unobserved": 0,
        "ignored": 1,
    }


def test_noncanonical_embedded_capture_is_ignored(tmp_path: Path) -> None:
    record = _record(0, 0)
    record["ng_trajectory"]["turns"][0]["model_calls"] = []
    record["ng_model_call_capture"] = {"calls": [_call()]}
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    result = run_health_checks(rollout_path, workers=1)

    [digest] = result.rollouts
    assert not digest.capture_observed
    assert digest.model_calls == 0
    assert set(digest.unobserved) == NO_CANONICAL_CALL_CHECKS
    assert digest.verdict == "unobserved"


def test_turn_without_a_model_call_reference_is_not_a_token_count_failure(tmp_path: Path) -> None:
    record = _record(0, 0, usage={"input_tokens": 3, "output_tokens": 2})
    record["ng_trajectory"]["turns"].append(
        {
            "invocation_id": "root",
            "task_id": "0",
            "rollout_id": "0-0",
            "turn_no": 2,
            "timestamp": 2.0,
            "answer": "second answer",
            "step_count": 2,
            "model_calls": [],
        }
    )
    rollout_path = _write_fixture(tmp_path, [(record, [_call()])])

    result = run_health_checks(rollout_path, workers=1)

    [digest] = result.rollouts
    assert digest.verdict == "healthy"
    assert digest.unobserved == []
    assert not any(finding.check == "model_call_missing_token_counts" for finding in digest.findings)


def test_bound_call_without_token_counts_is_a_model_call_finding(tmp_path: Path) -> None:
    rollout_path = _write_fixture(tmp_path, [(_record(0, 0), [_call(tokens_out=None)])])

    result = run_health_checks(rollout_path, workers=1)

    findings = [
        finding for finding in result.rollouts[0].findings if finding.check == "model_call_missing_token_counts"
    ]
    assert len(findings) == 1
    assert findings[0].locator == {"call_id": "c1"}
    assert findings[0].detail == {"missing": ["completion_tokens"]}


def test_current_producer_response_reference_binds_to_captured_call_id(tmp_path: Path) -> None:
    model_ref = {"type": "responses_api_models", "name": "openai_model"}
    row = {"_ng_task_index": 0, "_ng_rollout_index": 0}
    response = {
        "output": [{"type": "message", "role": "assistant", "content": "ok"}],
        "usage": {"input_tokens": 3, "output_tokens": 2},
    }
    result = {
        "response": response,
        "ng_trajectory": {
            "task_id": "0",
            "rollout_id": "0-0",
            "turns": [
                {
                    "invocation_id": "root",
                    "task_id": "0",
                    "rollout_id": "0-0",
                    "turn_no": 1,
                    "timestamp": 1.0,
                    "answer": "ok",
                    "step_count": 1,
                    "model_calls": [{"model_ref": model_ref, "response_id": "resp-1"}],
                }
            ],
        },
        "ng_model_call_capture": {
            "calls": [
                {
                    "model_call_id": "capture-uuid",
                    "model_ref": model_ref,
                    "response_id": "resp-1",
                    "status_code": 200,
                    "response_status": "completed",
                    "finish_reason": "stop",
                    "tokens_in": 3,
                    "tokens_out": 2,
                    "response": {"output_text": "ok"},
                }
            ]
        },
    }
    trajectory = rollout_collection._build_trajectory_record(row, result)
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(
        orjson.dumps(
            {**row, "response": response, "ng_trajectory": trajectory.model_dump(mode="json")},
            option=orjson.OPT_APPEND_NEWLINE,
        )
    )

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert trajectory.turns[0].model_calls[0].model_call_id is None
    assert trajectory.model_calls[0].model_call_id == "capture-uuid"
    assert digest.verdict == "healthy"
    assert digest.findings == []
    assert digest.unobserved == []


def test_correspondence_reports_only_explicit_canonical_contradictions(tmp_path: Path) -> None:
    model_ref = {"type": "responses_api_models", "name": "model"}
    record = _record(
        0,
        0,
        refs=[
            {"model_call_id": "missing"},
            {"model_ref": model_ref, "response_id": "duplicate-response"},
        ],
        usage={"input_tokens": 99, "output_tokens": 99},
    )
    rollout_path = _write_fixture(
        tmp_path,
        [
            (
                record,
                [
                    _call(model_call_id="failed", response_id="failed-response", status_code=500),
                    _call(model_call_id="duplicate-1", model_ref=model_ref, response_id="duplicate-response"),
                    _call(model_call_id="duplicate-2", model_ref=model_ref, response_id="duplicate-response"),
                ],
            )
        ],
    )

    result = run_health_checks(rollout_path, workers=1)

    kinds = {
        finding.detail.get("kind")
        for finding in result.rollouts[0].findings
        if finding.check == "trajectory_capture_mismatch"
    }
    assert kinds == {"missing_captured_call", "duplicated_captured_call"}
    assert {
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "rollout_token_count_mismatch",
        "model_call_runaway_generation",
    } <= set(result.rollouts[0].unobserved)
    assert result.summary["run"]["stats"]["duplicated_calls"] == {"replayed": 0, "rollouts": 0}
    assert health_checks._call_identity({"response_id": "loose"}) == "response::loose"
    assert health_checks._call_identity({}) is None


def test_correspondence_deduplicates_findings_for_one_call_identity(tmp_path: Path) -> None:
    record = _record(
        0,
        0,
        refs=[
            {"model_call_id": "c1", "response_id": "stale-1"},
            {"model_call_id": "c1", "response_id": "stale-2"},
        ],
    )
    rollout_path = _write_fixture(tmp_path, [(record, [_call()])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    mismatches = [finding for finding in digest.findings if finding.check == "trajectory_capture_mismatch"]
    assert len(mismatches) == 1
    assert mismatches[0].locator == {"call_id": "c1"}
    assert mismatches[0].detail == {"kind": "missing_captured_call"}


def test_correspondence_uses_bound_calls_and_gym_ids_for_replay(tmp_path: Path) -> None:
    record = _record(0, 0, usage={"input_tokens": 3, "output_tokens": 2})
    rollout_path = _write_fixture(
        tmp_path,
        [
            (
                record,
                [
                    _call(model_call_id="c1", response_id="placeholder"),
                    _call(
                        model_call_id="auxiliary",
                        response_id="placeholder",
                        tokens_in=100,
                        tokens_out=50,
                    ),
                ],
            )
        ],
    )

    result = run_health_checks(rollout_path, workers=1)

    correspondence = [
        finding for finding in result.rollouts[0].findings if finding.check == "trajectory_capture_mismatch"
    ]
    assert not correspondence
    assert result.summary["run"]["stats"]["duplicated_calls"] == {"replayed": 0, "rollouts": 0}


def test_partial_binding_checks_matched_calls_without_claiming_complete_accounting(tmp_path: Path) -> None:
    record = _record(
        0,
        0,
        refs=[{"model_call_id": "c1"}, {"model_call_id": "missing"}],
        usage={"input_tokens": 3, "output_tokens": 2},
    )
    rollout_path = _write_fixture(tmp_path, [(record, [_call()])])

    result = run_health_checks(rollout_path, workers=1)

    [digest] = result.rollouts
    assert any(finding.check == "trajectory_capture_mismatch" for finding in digest.findings)
    assert "rollout_token_count_mismatch" in digest.unobserved
    assert "model_call_missing_token_counts" not in digest.unobserved
    assert not any(finding.check == "model_call_missing_token_counts" for finding in digest.findings)


def test_call_failures_and_token_mismatches_have_separate_check_ids(tmp_path: Path) -> None:
    rollout_path = _write_fixture(
        tmp_path,
        [
            (
                _record(0, 0, usage={"input_tokens": 99, "output_tokens": 99}),
                [_call(status_code=500, error_category="upstream")],
            )
        ],
    )

    result = run_health_checks(rollout_path, workers=1)

    checks = [finding.check for finding in result.rollouts[0].findings]
    assert checks.count("rollout_ended_on_failed_model_call") == 1
    assert checks.count("rollout_token_count_mismatch") == 1
    assert "trajectory_capture_mismatch" not in checks
    failed = next(
        finding for finding in result.rollouts[0].findings if finding.check == "rollout_ended_on_failed_model_call"
    )
    assert failed.detail == {
        "status": 500,
        "error_category": "upstream",
        "model_ref": None,
        "observed_calls": 1,
        "successful_calls": 0,
    }


@pytest.mark.parametrize("gap", [None, "tool_timing_unavailable", "turn_model_call_scope_incomplete"])
def test_turn_call_scope_gap_only_gates_completeness_checks(tmp_path: Path, gap: str | None) -> None:
    # The reported usage includes an invocation-owned helper that is not a task turn.
    record = _record(0, 0, answer=None, usage={"input_tokens": 8, "output_tokens": 5})
    record["ng_trajectory"]["invocations"] = [
        {
            "kind": "agent_invocation",
            "invocation_id": "root",
            "model_calls": [{"model_call_id": "c1"}, {"model_call_id": "helper"}],
        }
    ]
    if gap is not None:
        record["ng_trajectory"]["gaps"] = [{"code": gap, "invocation_id": "root"}]
    rollout_path = _write_fixture(
        tmp_path,
        [
            (
                record,
                [
                    _call(status_code=500, error_category="upstream"),
                    _call(model_call_id="helper", response_id="helper-response", tokens_in=5, tokens_out=3),
                ],
            )
        ],
    )

    result = run_health_checks(rollout_path, workers=1)
    [digest] = result.rollouts
    checks = {finding.check for finding in digest.findings}
    incomplete = gap == "turn_model_call_scope_incomplete"
    assert set(digest.unobserved) == ({"rollout_token_count_mismatch"} if incomplete else set())
    assert "rollout_token_count_mismatch" not in checks
    assert "agent_turn_hollow" in checks
    assert digest.policy_calls_observed is not incomplete
    coverage = result.summary["run"]["artifacts"]["coverage"]["task_no_successful_model_calls"]
    assert coverage == {"evaluated": int(not incomplete), "unobserved": int(incomplete), "ignored": 0}
    assert ("task_no_successful_model_calls" in result.summary["tasks"]["0"]["flags"]) is not incomplete


def test_runs_that_reuse_task_indices_are_not_duplicates(tmp_path: Path) -> None:
    """`gym eval aggregate` merges runs from different environment servers, and each numbers its tasks from 0."""
    paths = []
    for run in ("run1", "run2"):
        (tmp_path / run).mkdir()
        _inventory(tmp_path / run, 1)
        paths.append(
            _write_fixture(tmp_path / run, [(_record(0, 0, usage={"input_tokens": 3, "output_tokens": 2}), [_call()])])
        )

    result = run_health_checks(paths, workers=1)

    assert result.summary["run"]["issues"]["rollout_duplicate_identity"] == 0
    assert all("rollout_duplicate_identity" not in {finding.check for finding in d.findings} for d in result.rollouts)
    artifacts = result.summary["run"]["artifacts"]
    assert (artifacts["expected"], artifacts["missing"], artifacts["healthy"]) == (2, 0, True)


def test_duplicate_rollout_identity_counts_once_at_task_scope(tmp_path: Path) -> None:
    duplicate = _record(7, 0, usage={"input_tokens": 3, "output_tokens": 2})
    rollout_path = _write_fixture(
        tmp_path,
        [
            (duplicate, [_call()]),
            (deepcopy(duplicate), [_call()]),
        ],
    )

    result = run_health_checks(rollout_path, workers=1)

    assert result.summary["run"]["verdicts"] == {"healthy": 0, "unhealthy": 2, "unobserved": 0}
    assert result.summary["run"]["issues"]["rollout_duplicate_identity"] == 2
    assert all(
        [finding.check for finding in digest.findings] == ["rollout_duplicate_identity"]
        and digest.findings[0].detail == {"duplicate_count": 2}
        for digest in result.rollouts
    )
    assert result.summary["tasks"]["7"] == {
        "repeats": 1,
        "healthy": 0,
        "unhealthy": 1,
        "unobserved": 0,
        "flags": [],
    }
    assert result.summary["run"]["artifacts"]["coverage"]["task_consistently_unhealthy"] == {
        "evaluated": 0,
        "unobserved": 1,
        "ignored": 0,
    }

    ignored = run_health_checks(rollout_path, workers=1, ignored_checks=["rollout_duplicate_identity"])
    assert ignored.summary["run"]["verdicts"] == {"healthy": 2, "unhealthy": 0, "unobserved": 0}
    assert ignored.summary["run"]["issues"]["rollout_duplicate_identity"] == 0
    assert ignored.summary["tasks"]["7"]["repeats"] == 1


def test_duplicate_rollout_identity_is_scoped_to_stage(tmp_path: Path) -> None:
    stage_0 = {**_record(7, 0, usage={"input_tokens": 3, "output_tokens": 2}), "stage_index": 0}
    stage_1 = {**deepcopy(stage_0), "stage_index": 1}
    rollout_path = _write_fixture(
        tmp_path,
        [
            (stage_0, [_call()]),
            (stage_1, [_call()]),
            (deepcopy(stage_1), [_call()]),
        ],
    )

    result = run_health_checks(rollout_path, workers=1)

    flagged = {
        digest.stage_index: [finding.check for finding in digest.findings]
        for digest in result.rollouts
        if digest.findings
    }
    assert flagged == {1: ["rollout_duplicate_identity"]}
    assert result.summary["run"]["verdicts"] == {"healthy": 1, "unhealthy": 2, "unobserved": 0}
    assert result.summary["run"]["issues"]["rollout_duplicate_identity"] == 2
    verdict_rows = [json.loads(line) for line in result.verdicts_path.read_text().splitlines()]
    assert [(row["stage_index"], row["verdict"]) for row in verdict_rows] == [
        (0, "healthy"),
        (1, "unhealthy"),
        (1, "unhealthy"),
    ]


def test_zero_token_call_is_flagged_and_nonempty_length_response_is_exempt(tmp_path: Path) -> None:
    rollout_path = _write_fixture(
        tmp_path,
        [
            (
                _record(0, 0, usage={"input_tokens": 3, "output_tokens": 0}),
                [
                    _call(
                        tokens_out=0,
                        finish_reason="length",
                        response={"choices": [{"message": {"content": "kept"}}]},
                    )
                ],
            )
        ],
    )

    result = run_health_checks(rollout_path, workers=1)

    checks = {finding.check for finding in result.rollouts[0].findings}
    assert "model_call_zero_completion_tokens" in checks
    assert "model_call_runaway_generation" not in checks
    assert health_checks._response_has_usable_output("malformed") is False
    assert health_checks._response_has_usable_output({"content": "visible"}) is True


@pytest.mark.parametrize(
    ("response", "expected_reason"),
    [
        (
            {
                "id": "r1",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output": [],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
            "max_output_tokens",
        ),
        (
            {
                "id": "r1",
                "choices": [{"finish_reason": "length", "message": {"content": ""}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 2},
            },
            "length",
        ),
        (
            {
                "id": "r1",
                "stop_reason": "max_tokens",
                "content": [],
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
            "max_tokens",
        ),
    ],
)
def test_current_capture_length_limit_reasons_trigger_runaway_generation(
    tmp_path: Path, response: dict, expected_reason: str
) -> None:
    call = build_model_call_record(
        {
            "model_call_id": "c1",
            "model_ref": {"type": "responses_api_models", "name": "openai_model"},
            "status_code": 200,
            "response": response,
        },
        call_index=0,
    ).model_dump(mode="json")
    rollout_path = _write_fixture(
        tmp_path,
        [(_record(0, 0, usage={"input_tokens": 3, "output_tokens": 2}), [call])],
    )

    [digest] = run_health_checks(rollout_path, workers=1).rollouts
    [finding] = [item for item in digest.findings if item.check == "model_call_runaway_generation"]

    assert call["finish_reason"] == expected_reason
    assert finding.detail == {"finish_reason": expected_reason}


_REASONING_ONLY = {
    "responses": {"output": [{"type": "reasoning", "summary": [{"type": "summary_text", "text": "partial"}]}]},
    "chat": {"choices": [{"message": {"content": "", "reasoning_content": "partial"}}]},
    "messages": {"content": [{"type": "thinking", "thinking": "partial"}]},
}
_ACTIONABLE = {
    "responses_function_call": {
        "output": [
            {"type": "reasoning", "summary": [{"type": "summary_text", "text": "partial"}]},
            {"type": "function_call", "call_id": "t1", "name": "run", "arguments": "{}"},
        ]
    },
    "responses_answer": {"output": [{"type": "message", "content": [{"type": "output_text", "text": "cut off"}]}]},
    "chat_tool_call": {"choices": [{"message": {"content": "", "tool_calls": [{"id": "t1", "type": "function"}]}}]},
    "messages_tool_use": {"content": [{"type": "thinking", "thinking": "partial"}, {"type": "tool_use", "id": "t1"}]},
}


@pytest.mark.parametrize(
    ("response", "flagged"),
    [
        *((response, True) for response in _REASONING_ONLY.values()),
        *((response, False) for response in _ACTIONABLE.values()),
    ],
    ids=[*_REASONING_ONLY, *_ACTIONABLE],
)
def test_a_length_limited_call_with_only_reasoning_is_a_runaway_generation(
    tmp_path: Path, response: dict, flagged: bool
) -> None:
    """Reasoning gives the agent nothing to act on; an answer or a tool call does. A later call's recovery does not
    erase the clipped one."""
    clipped = _call(
        model_call_id="clipped", response_id="r-clipped", started_at=1.0, finish_reason="length", response=response
    )
    recovered = _call(
        model_call_id="recovered",
        response_id="r-recovered",
        started_at=2.0,
        response={"output": [{"type": "message", "content": [{"type": "output_text", "text": "done"}]}]},
    )
    record = _record(
        0,
        0,
        refs=[{"model_call_id": "clipped"}, {"model_call_id": "recovered"}],
        usage={"input_tokens": 6, "output_tokens": 4},
    )
    rollout_path = _write_fixture(tmp_path, [(record, [clipped, recovered])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    flagged_calls = [f.locator["call_id"] for f in digest.findings if f.check == "model_call_runaway_generation"]
    assert flagged_calls == (["clipped"] if flagged else [])


def test_runaway_generation_is_unobserved_without_a_saved_response(tmp_path: Path) -> None:
    rollout_path = _write_fixture(
        tmp_path,
        [
            (
                _record(0, 0, usage={"input_tokens": 3, "output_tokens": 2}),
                [_call(finish_reason="length", response=None)],
            )
        ],
    )

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert "model_call_runaway_generation" in digest.unobserved
    assert not any(finding.check == "model_call_runaway_generation" for finding in digest.findings)


def test_malformed_records_and_check_failures_become_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_bytes(b"\n[]\n")
    parsed = run_health_checks(malformed, workers=1)
    [digest] = parsed.rollouts
    assert digest.task_index == "__unreadable_record__:input-0:line-2"
    assert digest.rollout_index == 0
    assert digest.verdict == "unhealthy"
    assert len(digest.findings) == 1
    assert digest.findings[0].check == "record_unreadable"
    assert digest.findings[0].locator == {"source_file": str(malformed), "line": 2}
    assert digest.findings[0].detail["reason"] == "rollout record is unreadable"
    assert set(digest.unobserved) == {
        "check_execution_error",
        "rollout_duplicate_identity",
        "rollout_missing_agent_turns",
        "rollout_ended_on_failed_model_call",
        "agent_turn_hollow",
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "trajectory_capture_mismatch",
        "rollout_token_count_mismatch",
        "model_call_runaway_generation",
    }

    # A malformed line must not be grouped with a valid rollout whose numeric
    # task index happens to equal the malformed line's global ordinal.
    collision = tmp_path / "collision.jsonl"
    collision.write_bytes(orjson.dumps(_record(1, 0), option=orjson.OPT_APPEND_NEWLINE) + b"[]\n")
    collision_result = run_health_checks(collision, workers=1)
    assert {digest.task_index for digest in collision_result.rollouts} == {
        1,
        "__unreadable_record__:input-0:line-2",
    }
    assert set(collision_result.summary["tasks"]) == {"1", "__unreadable_record__:input-0:line-2"}

    healthy = tmp_path / "healthy.jsonl"
    healthy.write_bytes(orjson.dumps(_record(0, 0), option=orjson.OPT_APPEND_NEWLINE))

    def broken_check(*args, **kwargs):
        raise TypeError("bad shape")

    monkeypatch.setitem(health_checks._ROLLOUT_CHECKS, "rollout_missing_agent_turns", broken_check)
    checked = run_health_checks(healthy, workers=1)
    finding = next(item for item in checked.rollouts[0].findings if item.check == "check_execution_error")
    assert finding.detail == {
        "reason": "check raised an unexpected exception",
        "failed_check": "rollout_missing_agent_turns",
        "error": "TypeError",
    }
    assert "rollout_missing_agent_turns" in checked.rollouts[0].unobserved
    assert not any(item.check == "record_unreadable" for item in checked.rollouts[0].findings)


def test_process_pool_success_path_and_explicit_rollout_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    class InlinePool:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, items, *, chunksize):
            assert chunksize == 1
            return map(function, items)

    monkeypatch.setattr(health, "ProcessPoolExecutor", InlinePool)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rollout_path = run_dir / "rollouts.jsonl"
    rollout_path.write_bytes(
        b"\n"
        + orjson.dumps(_record(0, 0), option=orjson.OPT_APPEND_NEWLINE)
        + orjson.dumps(_record(1, 0), option=orjson.OPT_APPEND_NEWLINE)
    )

    result = health.health_check_run_dir(run_dir, workers=2)

    assert len(result.rollouts) == 2
    assert "2 checked" in capsys.readouterr().out

    custom_path = run_dir / "custom-name.jsonl"
    rollout_path.rename(custom_path)
    file_result = health.health_check_run_dir(run_dir, rollout_file="custom-name.jsonl", workers=1)
    assert len(file_result.rollouts) == 2


@pytest.mark.parametrize(
    ("item_count", "workers", "expected"),
    [
        (1, 8, 1),
        (350, 8, 11),
        (1_600, 8, 50),
        (6_000, 8, 128),
        (30_000, 8, 128),
        (1_600, 16, 25),
    ],
)
def test_process_pool_chunksize(item_count: int, workers: int, expected: int) -> None:
    assert health._process_pool_chunksize(item_count, workers) == expected


def test_input_validation_and_nonstandard_filename_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_health_checks([], workers=1)
    with pytest.raises(FileNotFoundError, match="Rollout JSONL"):
        run_health_checks(tmp_path / "missing.jsonl", workers=1)
    with pytest.raises(FileNotFoundError, match="Run directory"):
        health.health_check_run_dir(tmp_path / "missing-run", workers=1)

    run_dir = tmp_path / "nonstandard"
    run_dir.mkdir()
    (run_dir / "a.jsonl").write_text("{}\n")
    (run_dir / "b.jsonl").write_text("{}\n")
    with pytest.raises(FileNotFoundError, match="rollouts.jsonl"):
        health.health_check_run_dir(run_dir, workers=1)
    explicit = health.health_check_run_dir(run_dir, rollout_file="a.jsonl", workers=1)
    assert len(explicit.rollouts) == 1

    one = tmp_path / "one.jsonl"
    one.write_text("{}\n")
    with pytest.raises(ValueError, match="workers"):
        run_health_checks(one, workers=0)
    with pytest.raises(ValueError, match="Unknown rollout health check.*not_a_check"):
        run_health_checks(one, ignored_checks=["not_a_check"], workers=1)


def test_health_check_config_accepts_csv_and_rejects_unknown_ids(tmp_path: Path) -> None:
    config = RolloutCollectionConfig(
        input_jsonl_fpath=str(tmp_path / "input.jsonl"),
        output_jsonl_fpath=str(tmp_path / "output.jsonl"),
        upload_rollouts=False,
        health_check_ignored_checks="model_call_missing_token_counts, model_call_zero_completion_tokens",
    )
    assert config.health_check_ignored_checks == [
        "model_call_missing_token_counts",
        "model_call_zero_completion_tokens",
    ]

    with pytest.raises(ValueError, match="Unknown rollout health check.*not_a_check"):
        RolloutCollectionConfig(
            input_jsonl_fpath=str(tmp_path / "input.jsonl"),
            output_jsonl_fpath=str(tmp_path / "output.jsonl"),
            upload_rollouts=False,
            health_check_ignored_checks=["not_a_check"],
        )


def _unreferenced_failure_record(calls: list[dict]) -> dict:
    """A rollout whose calls failed: nothing references them, because a failed
    call comes back with no response id for a producer to reference."""
    record = _record(0, 0, include_turn=False, include_response_output=False)
    record["ng_trajectory"]["invocations"] = [
        {"kind": "agent_invocation", "invocation_id": "root", "status": "incomplete", "model_calls": []}
    ]
    record["ng_trajectory"]["gaps"] = [{"code": "turns_unavailable"}]
    return record


def test_a_rollout_whose_last_model_call_failed_is_unhealthy(tmp_path: Path) -> None:
    """The case every bound-call check is blind to.

    A 403 carries no response id, so it is absent from `matched_calls` and
    bound-call checks cannot see it. Measured on a real run: six `403 auth`
    rows in the capture reported as `0 healthy, 0 unhealthy, 2 unobserved` with
    reward 0.0 -- indistinguishable from a model that scored zero.
    """
    record = _unreferenced_failure_record([])
    calls = [
        _call(call_index=index, model_call_id=None, response_id=None, status_code=403, error_category="auth")
        for index in range(3)
    ]
    rollout_path = _write_fixture(tmp_path, [(record, calls)])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert digest.verdict == "unhealthy"
    [finding] = [item for item in digest.findings if item.check == "rollout_ended_on_failed_model_call"]
    assert finding.detail["status"] == 403
    assert finding.detail["error_category"] == "auth"
    assert finding.detail["observed_calls"] == 3
    assert finding.detail["successful_calls"] == 0
    assert "rollout_ended_on_failed_model_call" not in digest.unobserved


def test_a_failure_the_client_retried_successfully_stays_healthy(tmp_path: Path) -> None:
    """Positive control for the previous test.

    The capture records every HTTP attempt, so a transient 429 the SDK retried
    appears as its own row. Judging any failed call would turn ordinary retries
    unhealthy; only the rollout ENDING on a failure is the signal.
    """
    record = _unreferenced_failure_record([])
    calls = [
        _call(call_index=0, model_call_id=None, response_id=None, status_code=429, error_category="rate_limit"),
        _call(call_index=1, model_call_id=None, response_id=None),
    ]
    rollout_path = _write_fixture(tmp_path, [(record, calls)])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert not [item for item in digest.findings if item.check == "rollout_ended_on_failed_model_call"]


_NO_GENERATION = {"response_status": None, "finish_reason": None}
FAILED_ATTEMPTS = {
    # Terminus cancels a call at its per-call limit: no status, no body, no usage.
    "cancelled": {"status_code": None, "error_category": "cancelled", "response": None, **_NO_GENERATION},
    # An HTTP error keeps its JSON error body, so "no response" alone would miss it.
    "http_429": {
        "status_code": 429,
        "error_category": "rate_limit",
        "response": {"error": {"code": 429}},
        **_NO_GENERATION,
    },
    "http_500": {
        "status_code": 500,
        "error_category": "upstream_error",
        "response": {"error": {"code": 500}},
        **_NO_GENERATION,
    },
}


def _owned_attempts(*calls: dict) -> dict:
    """A rollout whose turn owns every attempt, with transcript usage from the successful call only."""
    refs = [{"model_call_id": call["model_call_id"]} for call in calls]
    return _record(0, 0, refs=refs, usage={"input_tokens": 3, "output_tokens": 2})


@pytest.mark.parametrize("tokens_out", [None, 0])
@pytest.mark.parametrize("failure", FAILED_ATTEMPTS.values(), ids=FAILED_ATTEMPTS.keys())
def test_a_failed_attempt_followed_by_a_successful_call_is_healthy(tmp_path, failure, tokens_out):
    """A failed attempt has no usable usage, so its counts say nothing about token accounting.

    Measured on a TB2.1 run: 115 rollouts recovered from a Terminus per-call cancel and were
    still unhealthy, only because the cancelled attempt had no token counts.
    """
    failed = _call(model_call_id="attempt", started_at=1.0, tokens_in=None, tokens_out=tokens_out, **failure)
    retried = _call(model_call_id="retry", started_at=2.0)
    rollout_path = _write_fixture(tmp_path, [(_owned_attempts(failed, retried), [failed, retried])])

    result = run_health_checks(rollout_path, workers=1)
    [digest] = result.rollouts

    assert digest.verdict == "healthy"
    assert not digest.findings
    assert not digest.unobserved
    assert digest.model_call_errors == 1
    # The skipped attempt's tokens are counted nowhere else, so the run reports it as unknown usage.
    assert digest.model_call_errors_usage_unknown == 1
    assert result.summary["run"]["stats"]["model_call_errors"]["usage_unknown"] == 1


@pytest.mark.parametrize("failure", FAILED_ATTEMPTS.values(), ids=FAILED_ATTEMPTS.keys())
def test_a_rollout_ending_on_a_failed_attempt_is_unhealthy_for_that_reason_only(tmp_path, failure):
    succeeded = _call(model_call_id="first", started_at=1.0)
    failed = _call(model_call_id="last", started_at=2.0, tokens_in=None, tokens_out=None, **failure)
    rollout_path = _write_fixture(tmp_path, [(_owned_attempts(succeeded, failed), [succeeded, failed])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert digest.verdict == "unhealthy"
    assert {finding.check for finding in digest.findings} == {"rollout_ended_on_failed_model_call"}


# The two shapes a context-length overflow takes in the capture.
CONTEXT_OVERFLOWS = {
    # vLLM answers locally with an empty "length" completion when overflow errors are not propagated.
    "answered_locally": {
        "status_code": 200,
        "upstream_status_code": 400,
        "upstream_attempted": True,
        "error_category": "context_length_exceeded",
        "local_response_reason": "context_length_exceeded",
        "response_source": "local",
        "response_status": None,
        "finish_reason": "length",
    },
    # Propagated to the client as the upstream's own 400.
    "propagated": {
        "status_code": 400,
        "upstream_status_code": 400,
        "upstream_attempted": True,
        "error_category": "context_length_exceeded",
        "response_source": "upstream",
        "response": {"error": {"code": "context_length_exceeded"}},
        **_NO_GENERATION,
    },
}


# The adapter's category when the request's own output budget exceeds the whole window: no prompt fits.
INVALID_OUTPUT_BUDGETS = {
    name: overflow
    | {"error_category": "invalid_output_token_parameter"}
    | ({"local_response_reason": "invalid_output_token_parameter"} if "local_response_reason" in overflow else {})
    for name, overflow in CONTEXT_OVERFLOWS.items()
}


_POLICY_REF = {"type": "responses_api_models", "name": "policy_model"}
# A turn points at its call by the call id, or, as Terminal-Bench 4's turns do, by the model and response id.
TURN_CALL_REFERENCES = {
    "by_call_id": {"model_call_id": "c1"},
    "by_response_id": {"model_ref": _POLICY_REF, "response_id": "r1"},
}


@pytest.mark.parametrize("reference", TURN_CALL_REFERENCES.values(), ids=TURN_CALL_REFERENCES.keys())
@pytest.mark.parametrize("overflow", CONTEXT_OVERFLOWS.values(), ids=CONTEXT_OVERFLOWS.keys())
def test_an_empty_turn_from_a_context_overflow_is_not_hollow(tmp_path, overflow, reference):
    """The model answered, then filled its context: the empty turn is not hollow, and the rollout is not
    unhealthy."""
    answered = _call(model_call_id="c0", response_id="r0", started_at=1.0, model_ref=_POLICY_REF)
    record = _record(
        0,
        0,
        answer=None,
        refs=[{"model_call_id": "c0"}, reference],
        include_response_output=False,
        usage={"input_tokens": 3, "output_tokens": 2},
    )
    overflow_call = _call(
        **(
            {
                "call_index": 1,
                "started_at": 2.0,
                "tokens_in": None,
                "tokens_out": 0,
                "response": {},
                "model_ref": _POLICY_REF,
            }
            | overflow
        )
    )
    [digest] = run_health_checks(_write_fixture(tmp_path, [(record, [answered, overflow_call])]), workers=1).rollouts

    assert "agent_turn_hollow" not in {finding.check for finding in digest.findings}
    assert digest.verdict != "unhealthy"


def test_an_empty_turn_from_any_other_call_is_still_hollow(tmp_path):
    record = _record(0, 0, answer=None, include_response_output=False)
    empty_call = _call(tokens_out=0, response={})
    [digest] = run_health_checks(_write_fixture(tmp_path, [(record, [empty_call])]), workers=1).rollouts

    assert "agent_turn_hollow" in {finding.check for finding in digest.findings}


@pytest.mark.parametrize("overflow", CONTEXT_OVERFLOWS.values(), ids=CONTEXT_OVERFLOWS.keys())
def test_a_rollout_ending_on_a_context_overflow_is_not_flagged(tmp_path, overflow):
    """The model filled its context: a zero it earned, not an unhealthy rollout."""
    succeeded = _call(model_call_id="first", started_at=1.0)
    last = _call(model_call_id="last", started_at=2.0, tokens_in=None, tokens_out=0, **overflow)
    rollout_path = _write_fixture(tmp_path, [(_owned_attempts(succeeded, last), [succeeded, last])])

    result = run_health_checks(rollout_path, workers=1)
    [digest] = result.rollouts

    assert "rollout_ended_on_failed_model_call" not in {finding.check for finding in digest.findings}
    assert result.summary["run"]["stats"]["model_call_errors"]["ended_on_error"] == 0
    # The call is still a failed call in the error counts.
    assert digest.model_call_errors == 1


@pytest.mark.parametrize(
    "refusal",
    [INVALID_OUTPUT_BUDGETS["answered_locally"], CONTEXT_OVERFLOWS["answered_locally"]],
    ids=["invalid-budget", "context-overflow"],
)
@pytest.mark.parametrize("with_clipped", [True, False], ids=["beside-a-clipped-call", "alone"])
def test_a_length_refusal_without_a_saved_body_leaves_runaway_generation_observed(
    tmp_path: Path, refusal: dict, with_clipped: bool
) -> None:
    """The engine refused before generating anything, so a missing body is no missing evidence, and another call's
    runaway generation is still found."""
    calls = [
        _call(
            model_call_id="refusal",
            response_id="r-refusal",
            started_at=1.0,
            tokens_in=None,
            tokens_out=0,
            **(refusal | {"response": None}),
        )
    ]
    refs = [{"model_call_id": "refusal"}]
    if with_clipped:
        reasoning = {"type": "reasoning", "summary": [{"type": "summary_text", "text": "partial"}]}
        calls.append(
            _call(
                model_call_id="clipped",
                response_id="r-clipped",
                started_at=2.0,
                finish_reason="length",
                response={"output": [reasoning]},
            )
        )
        refs.append({"model_call_id": "clipped"})
    record = _record(0, 0, refs=refs, usage={"input_tokens": 6, "output_tokens": 4})
    rollout_path = _write_fixture(tmp_path, [(record, calls)])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert "model_call_runaway_generation" not in digest.unobserved
    flagged = [f.locator["call_id"] for f in digest.findings if f.check == "model_call_runaway_generation"]
    assert flagged == (["clipped"] if with_clipped else [])


@pytest.mark.parametrize("invalid", INVALID_OUTPUT_BUDGETS.values(), ids=INVALID_OUTPUT_BUDGETS.keys())
def test_a_budget_larger_than_the_window_still_flags_after_an_answer(tmp_path, invalid):
    """No prompt fits such a budget, so an earlier answer excuses nothing."""
    succeeded = _call(model_call_id="first", started_at=1.0)
    last = _call(model_call_id="last", started_at=2.0, tokens_in=None, tokens_out=0, **invalid)
    rollout_path = _write_fixture(tmp_path, [(_owned_attempts(succeeded, last), [succeeded, last])])

    result = run_health_checks(rollout_path, workers=1)
    [digest] = result.rollouts

    checks = {finding.check for finding in digest.findings}
    assert "rollout_ended_on_failed_model_call" in checks
    assert result.summary["run"]["stats"]["model_call_errors"]["ended_on_error"] == 1
    assert digest.model_call_errors == 1
    # The engine generated nothing, so the token checks have nothing to read, as for an overflow.
    assert not checks & {
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "model_call_runaway_generation",
    }


@pytest.mark.parametrize("ordered", [True, False], ids=["ordered", "unordered"])
@pytest.mark.parametrize("overflow", CONTEXT_OVERFLOWS.values(), ids=CONTEXT_OVERFLOWS.keys())
def test_an_overflow_before_the_model_ever_answered_still_flags(tmp_path, overflow, ordered):
    """With no answer first, the request never fit the deployment: a budget of 0 or one larger than the window,
    or a prompt too long for it. That is a broken run, not the model filling its context."""
    calls = [
        _call(
            call_index=index,
            model_call_id=None,
            response_id=None,
            tokens_out=0,
            **({"started_at": float(index)} if ordered else {}),
            **overflow,
        )
        for index in range(2)
    ]
    rollout_path = _write_fixture(tmp_path, [(_unreferenced_failure_record([]), calls)])

    result = run_health_checks(rollout_path, workers=1)
    [digest] = result.rollouts

    assert "rollout_ended_on_failed_model_call" in {finding.check for finding in digest.findings}
    assert result.summary["run"]["stats"]["model_call_errors"]["ended_on_error"] == 1


@pytest.mark.parametrize("failure", FAILED_ATTEMPTS.values(), ids=FAILED_ATTEMPTS.keys())
def test_a_failure_after_a_context_overflow_still_flags(tmp_path, failure):
    """Only the last call decides: an overflow earlier in the chain hides nothing that came after it."""
    overflow = _call(model_call_id="first", started_at=1.0, tokens_out=0, **CONTEXT_OVERFLOWS["propagated"])
    failed = _call(model_call_id="last", started_at=2.0, tokens_in=None, tokens_out=None, **failure)
    rollout_path = _write_fixture(tmp_path, [(_owned_attempts(overflow, failed), [overflow, failed])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert "rollout_ended_on_failed_model_call" in {finding.check for finding in digest.findings}


@pytest.mark.parametrize("failure", FAILED_ATTEMPTS.values(), ids=FAILED_ATTEMPTS.keys())
def test_a_context_overflow_after_only_failed_calls_still_flags(tmp_path, failure):
    """Every call failed, so the model never answered: the overflow says the request never fit."""
    failed = _call(model_call_id="first", started_at=1.0, tokens_in=None, tokens_out=None, **failure)
    overflow = _call(model_call_id="last", started_at=2.0, tokens_out=0, **CONTEXT_OVERFLOWS["propagated"])
    rollout_path = _write_fixture(tmp_path, [(_owned_attempts(failed, overflow), [failed, overflow])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert "rollout_ended_on_failed_model_call" in {finding.check for finding in digest.findings}


def test_another_models_answer_does_not_excuse_an_overflow(tmp_path):
    """Judged per model: a judge's answer says nothing of whether the policy's request fit."""
    judge = _call(
        model_call_id="judge", started_at=1.0, model_ref={"type": "responses_api_models", "name": "judge_model"}
    )
    overflow = _call(
        model_call_id="last",
        started_at=2.0,
        tokens_in=None,
        tokens_out=0,
        model_ref=_POLICY_REF,
        **CONTEXT_OVERFLOWS["propagated"],
    )
    rollout_path = _write_fixture(tmp_path, [(_owned_attempts(judge, overflow), [judge, overflow])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert "rollout_ended_on_failed_model_call" in {finding.check for finding in digest.findings}


def _conversation(dialect: str, opening: str, *messages: str) -> dict:
    """A chat or Responses request whose conversation opens with `opening`."""
    turns = [{"role": "user", "content": message} for message in messages]
    if dialect == "responses":
        return {"instructions": opening, "input": turns}
    return {"messages": [{"role": "system", "content": opening}, *turns]}


@pytest.mark.parametrize("dialect", ["chat", "responses"])
def test_an_overflow_after_an_answer_in_the_same_conversation_is_not_flagged(tmp_path, dialect):
    """A multi-turn agent answered, then its own conversation outgrew the window: its calls open the same way."""
    answered = _call(
        model_call_id="first",
        started_at=1.0,
        model_ref=_POLICY_REF,
        request=_conversation(dialect, "You are an agent.", "task"),
    )
    overflow = _call(
        model_call_id="last",
        started_at=2.0,
        tokens_in=None,
        tokens_out=0,
        model_ref=_POLICY_REF,
        request=_conversation(dialect, "You are an agent.", "task", "next step"),
        **CONTEXT_OVERFLOWS["propagated"],
    )
    rollout_path = _write_fixture(tmp_path, [(_owned_attempts(answered, overflow), [answered, overflow])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert "rollout_ended_on_failed_model_call" not in {finding.check for finding in digest.findings}


@pytest.mark.parametrize("dialect", ["chat", "responses"])
@pytest.mark.parametrize("opening", ["You play the customer.", None], ids=["another_conversation", "no_request"])
def test_an_answer_outside_the_overflowing_conversation_does_not_excuse_it(tmp_path, dialect, opening):
    """tau2 can serve its user simulator from the policy's own model server, and the simulator speaks first. Its
    answer says nothing of whether the agent's request fit, and neither does an answer whose request was not
    captured."""
    answered = _call(
        model_call_id="simulator",
        started_at=1.0,
        model_ref=_POLICY_REF,
        request=_conversation(dialect, opening, "hello") if opening else None,
    )
    overflow = _call(
        model_call_id="agent",
        started_at=2.0,
        tokens_in=None,
        tokens_out=0,
        model_ref=_POLICY_REF,
        request=_conversation(dialect, "You are an agent.", "task"),
        **CONTEXT_OVERFLOWS["propagated"],
    )
    rollout_path = _write_fixture(tmp_path, [(_unreferenced_failure_record([]), [answered, overflow])])

    result = run_health_checks(rollout_path, workers=1)
    [digest] = result.rollouts

    assert "rollout_ended_on_failed_model_call" in {finding.check for finding in digest.findings}
    assert result.summary["run"]["stats"]["model_call_errors"]["ended_on_error"] == 1


@pytest.mark.parametrize(
    "harness_sessions,flagged",
    [((None, None), True), (("shared", "shared"), False)],
    ids=["two_sessions", "two_activations_of_one_harness_session"],
)
def test_a_session_tells_apart_conversations_that_open_alike(tmp_path, harness_sessions, flagged):
    """Two sessions can share an agent's instructions: one session's answer says nothing of whether the other's first
    request fit. Activations of one harness session continue one conversation."""
    answered = _call(
        model_call_id="first",
        started_at=1.0,
        model_ref=_POLICY_REF,
        client_session_id="s1",
        request=_conversation("chat", "You are an agent.", "task A"),
    )
    overflow = _call(
        model_call_id="second",
        started_at=2.0,
        tokens_in=None,
        tokens_out=0,
        model_ref=_POLICY_REF,
        client_session_id="s2",
        request=_conversation("chat", "You are an agent.", "task B"),
        **CONTEXT_OVERFLOWS["propagated"],
    )
    record = _unreferenced_failure_record([])
    record["ng_trajectory"]["invocations"] = [
        {"kind": "agent_invocation", "invocation_id": session, "status": "completed"}
        | ({"harness_session_id": harness} if harness else {})
        for session, harness in zip(("s1", "s2"), harness_sessions)
    ]
    rollout_path = _write_fixture(tmp_path, [(record, [answered, overflow])])

    result = run_health_checks(rollout_path, workers=1)
    [digest] = result.rollouts

    assert ("rollout_ended_on_failed_model_call" in {finding.check for finding in digest.findings}) is flagged
    assert result.summary["run"]["stats"]["model_call_errors"]["ended_on_error"] == int(flagged)


def test_an_empty_overflow_turn_stays_hollow_when_no_other_check_accounts_for_it(tmp_path: Path) -> None:
    """Without timestamps, an overflow beside another conversation's answer cannot be ordered, so
    `rollout_ended_on_failed_model_call` says nothing: the empty turn is then the only finding left."""
    answered = _call(model_call_id="aux", model_ref=_POLICY_REF, request={"input": "another conversation"})
    overflow = _call(
        model_ref=_POLICY_REF, tokens_in=None, tokens_out=0, response={}, **CONTEXT_OVERFLOWS["answered_locally"]
    )
    record = _record(0, 0, answer=None, include_response_output=False)
    rollout_path = _write_fixture(tmp_path, [(record, [answered, overflow])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    checks = {finding.check for finding in digest.findings}
    assert "agent_turn_hollow" in checks
    assert "rollout_ended_on_failed_model_call" not in checks
    assert digest.verdict == "unhealthy"


def test_an_unordered_chain_of_an_overflow_and_a_transport_failure_still_flags(tmp_path: Path) -> None:
    calls = [
        _call(call_index=0, model_call_id=None, response_id=None, tokens_out=0, **CONTEXT_OVERFLOWS["propagated"]),
        _call(call_index=1, model_call_id=None, response_id=None, **FAILED_ATTEMPTS["cancelled"]),
    ]
    rollout_path = _write_fixture(tmp_path, [(_unreferenced_failure_record([]), calls)])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert "rollout_ended_on_failed_model_call" in {finding.check for finding in digest.findings}


def test_an_empty_retry_after_a_context_overflow_in_the_same_turn_is_still_hollow(tmp_path: Path) -> None:
    """Only the turn's last call can excuse it: a successful empty retry after an overflow is hollow."""
    overflow = _call(
        model_call_id="overflow",
        response_id="r-overflow",
        started_at=1.0,
        tokens_in=None,
        tokens_out=0,
        model_ref=_POLICY_REF,
        **CONTEXT_OVERFLOWS["propagated"],
    )
    retry = _call(
        model_call_id="retry",
        response_id="r-retry",
        call_index=1,
        started_at=2.0,
        tokens_out=1,
        response={},
        model_ref=_POLICY_REF,
    )
    record = _record(
        0,
        0,
        answer=None,
        refs=[{"model_call_id": "overflow"}, {"model_call_id": "retry"}],
        include_response_output=False,
        usage={"input_tokens": 3, "output_tokens": 1},
    )
    [digest] = run_health_checks(_write_fixture(tmp_path, [(record, [overflow, retry])]), workers=1).rollouts

    checks = {finding.check for finding in digest.findings}
    assert "agent_turn_hollow" in checks
    assert "rollout_ended_on_failed_model_call" not in checks
    assert digest.verdict == "unhealthy"


@pytest.mark.parametrize("failure", FAILED_ATTEMPTS.values(), ids=FAILED_ATTEMPTS.keys())
def test_an_empty_turn_ending_on_a_context_overflow_after_a_failed_call_is_not_hollow(tmp_path, failure):
    failed = _call(
        model_call_id="failed", response_id="r-failed", started_at=1.0, tokens_in=None, tokens_out=None, **failure
    )
    overflow = _call(
        model_call_id="overflow",
        response_id="r-overflow",
        call_index=1,
        started_at=2.0,
        tokens_in=None,
        tokens_out=0,
        model_ref=_POLICY_REF,
        **CONTEXT_OVERFLOWS["propagated"],
    )
    record = _record(
        0,
        0,
        answer=None,
        refs=[{"model_call_id": "failed"}, {"model_call_id": "overflow"}],
        include_response_output=False,
    )
    [digest] = run_health_checks(_write_fixture(tmp_path, [(record, [failed, overflow])]), workers=1).rollouts

    assert "agent_turn_hollow" not in {finding.check for finding in digest.findings}


def test_an_empty_turn_whose_own_reference_does_not_resolve_is_still_hollow(tmp_path: Path) -> None:
    """Turn 2 names the overflow call with a stale response id, so the call that turn 1 resolved excuses only turn 1."""
    overflow = _call(
        tokens_in=None, tokens_out=0, response={}, model_ref=_POLICY_REF, **CONTEXT_OVERFLOWS["answered_locally"]
    )
    record = _record(0, 0, answer=None, refs=[{"model_call_id": "c1"}], include_response_output=False)
    stale_turn = dict(record["ng_trajectory"]["turns"][0])
    stale_turn.update(turn_no=2, timestamp=2.0, model_calls=[{"model_call_id": "c1", "response_id": "stale"}])
    record["ng_trajectory"]["turns"].append(stale_turn)
    [digest] = run_health_checks(_write_fixture(tmp_path, [(record, [overflow])]), workers=1).rollouts

    hollow = [finding.locator for finding in digest.findings if finding.check == "agent_turn_hollow"]
    assert hollow == [{"turn": 2}]
    assert "trajectory_capture_mismatch" in {finding.check for finding in digest.findings}


def test_an_ambiguous_response_reference_is_not_excused_but_an_exact_call_reference_is(tmp_path: Path) -> None:
    """Two overflow calls share one model and response id: the call-id reference resolves, the response-id one
    matches both calls and excuses nothing."""
    shared = {"response_id": "shared", "tokens_in": None, "tokens_out": 0, "response": {}, "model_ref": _POLICY_REF}
    first = _call(model_call_id="c1", **shared, **CONTEXT_OVERFLOWS["answered_locally"])
    second = _call(model_call_id="c2", call_index=1, **shared, **CONTEXT_OVERFLOWS["answered_locally"])
    record = _record(0, 0, answer=None, refs=[{"model_call_id": "c1"}], include_response_output=False)
    ambiguous_turn = dict(record["ng_trajectory"]["turns"][0])
    ambiguous_turn.update(turn_no=2, timestamp=2.0, model_calls=[{"model_ref": _POLICY_REF, "response_id": "shared"}])
    record["ng_trajectory"]["turns"].append(ambiguous_turn)
    [digest] = run_health_checks(_write_fixture(tmp_path, [(record, [first, second])]), workers=1).rollouts

    hollow = [finding.locator for finding in digest.findings if finding.check == "agent_turn_hollow"]
    assert hollow == [{"turn": 2}]


def test_an_incomplete_capture_excuses_no_empty_turn(tmp_path: Path) -> None:
    """An incomplete capture leaves `rollout_ended_on_failed_model_call` unobserved: it cannot flag a request that
    never fit, so the empty turn is still hollow."""
    overflow = _call(
        tokens_in=None, tokens_out=0, response={}, model_ref=_POLICY_REF, **CONTEXT_OVERFLOWS["answered_locally"]
    )
    record = _record(0, 0, answer=None, refs=[{"model_call_id": "c1"}], include_response_output=False)
    record["ng_trajectory"]["gaps"] = [{"code": "model_call_capture_incomplete"}]
    [digest] = run_health_checks(_write_fixture(tmp_path, [(record, [overflow])]), workers=1).rollouts

    assert "rollout_ended_on_failed_model_call" in digest.unobserved
    assert "agent_turn_hollow" in {finding.check for finding in digest.findings}


def test_a_400_without_the_overflow_category_still_flags(tmp_path: Path) -> None:
    """The overflow is read from the recorded category, never inferred from a 400 alone."""
    succeeded = _call(model_call_id="first", started_at=1.0)
    rejected = _call(
        model_call_id="last",
        started_at=2.0,
        status_code=400,
        upstream_status_code=400,
        upstream_attempted=True,
        error_category="client_error",
        tokens_in=None,
        tokens_out=None,
        **_NO_GENERATION,
    )
    rollout_path = _write_fixture(tmp_path, [(_owned_attempts(succeeded, rejected), [succeeded, rejected])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert "rollout_ended_on_failed_model_call" in {finding.check for finding in digest.findings}


def test_a_body_capture_could_not_parse_still_needs_token_counts(tmp_path: Path) -> None:
    """`capture_parse_error` marks a 2xx call whose body was unreadable: a capture defect, not a failed call."""
    unparsed = _call(
        error_category="capture_parse_error", response=None, tokens_in=None, tokens_out=None, **_NO_GENERATION
    )
    rollout_path = _write_fixture(tmp_path, [(_record(0, 0), [unparsed])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert "model_call_missing_token_counts" in {finding.check for finding in digest.findings}
    assert digest.model_call_errors_usage_unknown == 0


def test_the_check_is_unobserved_when_no_calls_were_captured(tmp_path: Path) -> None:
    record = _unreferenced_failure_record([])
    rollout_path = _write_fixture(tmp_path, [(record, [])])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert "rollout_ended_on_failed_model_call" in digest.unobserved


def test_a_producer_reporting_no_turns_is_missing_turns_not_unobserved() -> None:
    """`rollout_missing_agent_turns` exists to catch a rollout with no turns,
    and was skipped in exactly that case: `turns_unavailable` was appended
    whenever the turn list was empty, and the AGENT_TURNS gate then dropped the
    check."""
    result = {
        "ng_trajectory": {
            "task_id": "0",
            "rollout_id": "0-0",
            "invocations": [{"kind": "agent_invocation", "invocation_id": "root", "status": "incomplete"}],
            "turns": [],
        }
    }
    trajectory = rollout_collection._build_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, result)

    assert "turns_unavailable" not in {gap.code for gap in trajectory.gaps}


def test_an_agent_that_publishes_no_trajectory_still_reports_turns_unavailable() -> None:
    """Regression guard: 45 of 47 agents publish no `TrajectoryRecord`. Their
    turns are genuinely unavailable, not empty, and must stay unobserved rather
    than turn the whole fleet unhealthy."""
    trajectory = rollout_collection._build_trajectory_record({"_ng_task_index": 0, "_ng_rollout_index": 0}, {})

    assert "turns_unavailable" in {gap.code for gap in trajectory.gaps}


def test_the_ended_on_error_statistic_and_the_finding_agree(tmp_path: Path) -> None:
    """The digest already reported `ended_on_error` for this rollout while the
    verdict stayed `unobserved` -- the gate knew and said nothing. Both now read
    one predicate, so they cannot drift apart again."""
    record = _unreferenced_failure_record([])
    calls = [_call(call_index=0, model_call_id=None, response_id=None, status_code=500)]
    rollout_path = _write_fixture(tmp_path, [(record, calls)])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert digest.ended_on_error is True
    assert "rollout_ended_on_failed_model_call" in {finding.check for finding in digest.findings}


def test_an_incomplete_capture_is_not_counted_as_ending_on_an_error(tmp_path: Path) -> None:
    """The check is unobserved on an incomplete capture, so the statistic leaves the rollout out as well."""
    succeeded = _call(model_call_id="first", started_at=1.0)
    failed = _call(
        model_call_id="last", started_at=2.0, tokens_in=None, tokens_out=None, **FAILED_ATTEMPTS["http_500"]
    )
    record = _owned_attempts(succeeded, failed)
    record["ng_trajectory"]["gaps"] = [{"code": "model_call_capture_incomplete"}]

    result = run_health_checks(_write_fixture(tmp_path, [(record, [succeeded, failed])]), workers=1)
    [digest] = result.rollouts

    assert "rollout_ended_on_failed_model_call" in digest.unobserved
    assert "rollout_ended_on_failed_model_call" not in {finding.check for finding in digest.findings}
    assert digest.ended_on_error is False
    assert result.summary["run"]["stats"]["model_call_errors"]["ended_on_error"] == 0


POLICY_REF = {"type": "responses_api_models", "name": "policy_model"}
JUDGE_REF = {"type": "responses_api_models", "name": "genrm_model"}


def _unreferenced(calls_note=None) -> dict:
    record = _record(0, 0, include_turn=False, include_response_output=False)
    record["ng_trajectory"]["invocations"] = [
        {"kind": "agent_invocation", "invocation_id": "root", "status": "incomplete", "model_calls": []}
    ]
    record["ng_trajectory"]["gaps"] = [{"code": "turns_unavailable"}]
    return record


def test_an_auxiliary_call_after_a_failed_policy_call_does_not_hide_it(tmp_path: Path) -> None:
    """A judge request landing after the policy failed is not the policy recovering.

    Judged across every captured call, the judge's 200 was simply last and the
    rollout read as fine.
    """
    calls = [
        _call(
            call_index=0,
            model_call_id=None,
            response_id=None,
            model_ref=POLICY_REF,
            started_at=1.0,
            status_code=403,
            error_category="auth",
        ),
        _call(
            call_index=1, model_call_id=None, response_id=None, model_ref=JUDGE_REF, started_at=2.0, status_code=200
        ),
    ]
    rollout_path = _write_fixture(tmp_path, [(_unreferenced(), calls)])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert digest.verdict == "unhealthy"
    [finding] = [f for f in digest.findings if f.check == "rollout_ended_on_failed_model_call"]
    assert finding.detail["status"] == 403
    assert finding.detail["model_ref"] == "responses_api_models/policy_model"


def test_a_retry_recorded_out_of_order_is_not_reported_as_a_failure(tmp_path: Path) -> None:
    """`call_index` is the merged trajectory position, not chronological.

    `_build_trajectory_record` puts producer-supplied calls first and appends
    unmatched capture rows, so a successful retry can sit BEFORE the attempt it
    retried. Judged on list order that reads as ending on a failure.
    """
    calls = [
        _call(
            call_index=0,
            model_call_id=None,
            response_id="r-retry",
            model_ref=POLICY_REF,
            started_at=2.0,
            status_code=200,
        ),
        _call(
            call_index=1, model_call_id=None, response_id=None, model_ref=POLICY_REF, started_at=1.0, status_code=500
        ),
    ]
    rollout_path = _write_fixture(tmp_path, [(_unreferenced(), calls)])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert not [f for f in digest.findings if f.check == "rollout_ended_on_failed_model_call"]
    assert digest.ended_on_error is False


def test_an_unorderable_chain_is_not_called_a_failure(tmp_path: Path) -> None:
    """Without timestamps a multi-call chain cannot be ordered; stay quiet."""
    calls = [
        _call(call_index=0, model_call_id=None, response_id="r1", model_ref=POLICY_REF, status_code=200),
        _call(call_index=1, model_call_id=None, response_id=None, model_ref=POLICY_REF, status_code=500),
    ]
    rollout_path = _write_fixture(tmp_path, [(_unreferenced(), calls)])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    assert not [f for f in digest.findings if f.check == "rollout_ended_on_failed_model_call"]


def test_each_model_that_ends_on_a_failure_is_reported(tmp_path: Path) -> None:
    calls = [
        _call(
            call_index=0,
            model_call_id=None,
            response_id=None,
            model_ref=POLICY_REF,
            started_at=1.0,
            status_code=403,
            error_category="auth",
        ),
        _call(
            call_index=1, model_call_id=None, response_id=None, model_ref=JUDGE_REF, started_at=2.0, status_code=500
        ),
    ]
    rollout_path = _write_fixture(tmp_path, [(_unreferenced(), calls)])

    [digest] = run_health_checks(rollout_path, workers=1).rollouts

    refs = sorted(f.detail["model_ref"] for f in digest.findings if f.check == "rollout_ended_on_failed_model_call")
    assert refs == ["responses_api_models/genrm_model", "responses_api_models/policy_model"]


@pytest.mark.parametrize("tokens", [None, 0])
@pytest.mark.parametrize("response", [None, {}])
@pytest.mark.parametrize("finish_reason", ["length", "max_output_tokens", "max_tokens"])
def test_recovered_explicit_overflow_does_not_fail_generation_checks(tmp_path, tokens, response, finish_reason):
    record = _record(0, 0, refs=[{"model_call_id": "rejected"}, {"model_call_id": "ok"}])
    rejected = _call(
        model_call_id="rejected",
        started_at=1,
        tokens_in=tokens,
        tokens_out=tokens,
        response=response,
        upstream_attempted=True,
        upstream_status_code=400,
        local_response_reason="context_length_exceeded",
        error_category="context_length_exceeded",
        finish_reason=finish_reason,
    )
    [digest] = run_health_checks(
        _write_fixture(tmp_path, [(record, [rejected, _call(model_call_id="ok", started_at=3)])]), workers=1
    ).rollouts
    assert digest.verdict != "unhealthy"
    assert not digest.findings
    assert digest.model_call_errors == 1
    assert not {
        "rollout_ended_on_failed_model_call",
        "model_call_missing_token_counts",
        "model_call_zero_completion_tokens",
        "model_call_runaway_generation",
    } & set(digest.unobserved)


@pytest.mark.parametrize(
    "mutation",
    [
        {"upstream_attempted": None},
        {"upstream_status_code": None},
        {"upstream_status_code": 500},
        {"local_response_reason": None, "error_category": "timeout"},
        {"local_response_reason": None, "error_category": None, "upstream_status_code": None},
    ],
)
def test_uncertain_or_real_generations_keep_quality_checks(tmp_path, mutation):
    record = _record(0, 0, refs=[{"model_call_id": "earlier"}, {"model_call_id": "ok"}])
    earlier = (
        _call(
            model_call_id="earlier",
            started_at=1,
            tokens_in=None,
            tokens_out=0,
            response={},
            finish_reason="length",
            upstream_attempted=True,
            upstream_status_code=400,
            local_response_reason="context_length_exceeded",
            error_category="context_length_exceeded",
        )
        | mutation
    )
    [digest] = run_health_checks(
        _write_fixture(tmp_path, [(record, [earlier, _call(model_call_id="ok", started_at=3)])]), workers=1
    ).rollouts
    assert {
        "model_call_missing_token_counts",
        "model_call_zero_completion_tokens",
        "model_call_runaway_generation",
    } <= {f.check for f in digest.findings}


def test_old_failed_check_ignore_name_maps_to_new_check(tmp_path):
    assert health.normalize_ignored_checks(["model_call_failed", "rollout_ended_on_failed_model_call"]) == (
        "rollout_ended_on_failed_model_call",
    )
    result = run_health_checks(
        _write_fixture(tmp_path, [(_record(0, 0), [_call(status_code=500)])]),
        workers=1,
        ignored_checks=["model_call_failed"],
    )
    assert result.summary["run"]["ignored_checks"] == ["rollout_ended_on_failed_model_call"]
    assert "rollout_ended_on_failed_model_call" not in {f.check for f in result.rollouts[0].findings}


@pytest.mark.parametrize("final_failed", [False, True])
def test_only_final_failure_affects_health_for_bound_calls(tmp_path, final_failed):
    record = _record(
        0,
        0,
        refs=[{"model_call_id": "first"}, {"model_call_id": "last"}],
        usage={"input_tokens": 6, "output_tokens": 4},
    )
    calls = [
        _call(model_call_id="first", started_at=1, status_code=500),
        _call(model_call_id="last", started_at=2, status_code=500 if final_failed else 200),
    ]
    [digest] = run_health_checks(_write_fixture(tmp_path, [(record, calls)]), workers=1).rollouts
    assert {finding.check for finding in digest.findings} == (
        {"rollout_ended_on_failed_model_call"} if final_failed else set()
    )
    assert digest.verdict == ("unhealthy" if final_failed else "healthy")
    assert digest.model_call_errors == (2 if final_failed else 1)


@pytest.mark.parametrize("duplicate_reference", [False, True])
def test_usage_counts_invocation_owned_compaction_once(tmp_path: Path, duplicate_reference: bool) -> None:
    record = _record(0, 0, usage={"input_tokens": 8, "output_tokens": 5})
    refs = [{"model_call_id": "c1"}, {"model_call_id": "summary"}]
    if duplicate_reference:
        refs.append({"model_call_id": "summary"})
    record["ng_trajectory"]["invocations"] = [{"invocation_id": "root", "model_calls": refs}]
    rollout_path = _write_fixture(
        tmp_path,
        [
            (
                record,
                [
                    _call(),
                    _call(model_call_id="summary", response_id="summary-response", tokens_in=5, tokens_out=3),
                    _call(model_call_id="unowned", response_id="unowned-response", tokens_in=100, tokens_out=100),
                ],
            )
        ],
    )
    [digest] = run_health_checks(rollout_path, workers=1).rollouts
    assert not digest.findings
    assert "rollout_token_count_mismatch" not in digest.unobserved


def test_usage_compaction_missing_capture_remains_unobserved(tmp_path: Path) -> None:
    record = _record(0, 0, usage={"input_tokens": 8, "output_tokens": 5})
    record["ng_trajectory"]["invocations"] = [
        {
            "invocation_id": "root",
            "model_calls": [
                {"model_call_id": "c1"},
                {"model_call_id": "missing-summary"},
            ],
        }
    ]
    [digest] = run_health_checks(_write_fixture(tmp_path, [(record, [_call()])]), workers=1).rollouts
    assert "rollout_token_count_mismatch" in digest.unobserved
    assert {f.check for f in digest.findings} == {"trajectory_capture_mismatch"}


def test_usage_still_reports_incorrect_invocation_total(tmp_path: Path) -> None:
    record = _record(0, 0, usage={"input_tokens": 8, "output_tokens": 6})
    record["ng_trajectory"]["invocations"] = [
        {
            "invocation_id": "root",
            "model_calls": [
                {"model_call_id": "c1"},
                {"model_call_id": "summary"},
            ],
        }
    ]
    calls = [_call(), _call(model_call_id="summary", response_id="summary-response", tokens_in=5, tokens_out=3)]
    [digest] = run_health_checks(_write_fixture(tmp_path, [(record, calls)]), workers=1).rollouts
    [finding] = digest.findings
    assert finding.check == "rollout_token_count_mismatch"
    assert finding.detail == {
        "transcript_prompt": 8,
        "transcript_completion": 6,
        "capture_prompt": 8,
        "capture_completion": 5,
    }


def test_summary_counts_expected_failed_and_missing_rollouts(tmp_path: Path) -> None:
    rollout_path = _write_fixture(tmp_path, [(_record(task, 0), [_call()]) for task in range(4)])
    (tmp_path / "rollouts_materialized_inputs.jsonl").write_bytes(
        b"".join(orjson.dumps({"_ng_task_index": task, "_ng_rollout_index": 0}) + b"\n" for task in range(8))
    )
    # Task 0 failed and then landed, so it is not failed; task 4's last attempt is the one that stands.
    failures = [(0, "agent_run_error"), (4, "agent_run_error"), (4, "agent_timeout"), (5, "agent_run_error")]
    failures += [(6, "judge_failed")]
    (tmp_path / "rollouts_failures.jsonl").write_bytes(
        b"".join(
            orjson.dumps({"_ng_task_index": task, "_ng_rollout_index": 0, "_ng_failure_class": kind}) + b"\n"
            for task, kind in failures
        )
        + b'{"_ng_task_index": 7'  # A killed writer's truncated tail is not a failure.
    )

    result = run_health_checks(rollout_path, workers=1)

    artifacts = json.loads(result.summary_path.read_text())["run"]["artifacts"]
    assert artifacts["records"] == 4
    assert artifacts["expected"] == 8
    assert artifacts["failed"] == {
        "total": 3,
        "final": 0,
        "not_final": 3,
        "by_class": {"agent_run_error": 1, "agent_timeout": 1, "judge_failed": 1},
    }
    assert artifacts["missing"] == 1


def _run_with_outcomes(
    root: Path, *, landed: int, not_final: int = 0, final: int = 0, missing: int = 0, **limits: float
) -> dict:
    """Run health on a run of 100 tasks: `landed` scored, then not_final and final failures, the rest owed with no row."""
    assert landed + not_final + final + missing == 100
    rollout_path = _write_fixture(root, [(_record(task, 0), [_call()]) for task in range(landed)])
    (root / "rollouts_materialized_inputs.jsonl").write_bytes(
        b"".join(orjson.dumps({"_ng_task_index": task, "_ng_rollout_index": 0}) + b"\n" for task in range(100))
    )
    failures = [(task, False) for task in range(landed, landed + not_final)]
    failures += [(task, True) for task in range(landed + not_final, landed + not_final + final)]
    (root / "rollouts_failures.jsonl").write_bytes(
        b"".join(
            orjson.dumps(
                {
                    "_ng_task_index": task,
                    "_ng_rollout_index": 0,
                    "_ng_failure_class": "agent_run_error",
                    "_ng_failure_terminal": terminal,
                }
            )
            + b"\n"
            for task, terminal in failures
        )
    )
    from nemo_gym.health.types import RunHealthLimits

    result = run_health_checks(rollout_path, workers=1, limits=RunHealthLimits(**limits))
    return json.loads(result.summary_path.read_text())["run"]["artifacts"]


def test_run_is_healthy_when_nothing_is_missing_and_failures_stay_below_the_limits(tmp_path: Path) -> None:
    artifacts = _run_with_outcomes(tmp_path, landed=98, not_final=1, final=1)

    assert (artifacts["failed"]["not_final"], artifacts["failed"]["final"]) == (1, 1)
    assert artifacts["healthy"] is True
    assert artifacts["unhealthy_reasons"] == []


def test_failures_have_no_limit_unless_one_is_configured(tmp_path: Path) -> None:
    """The failure limits are off by default: the counts are written, and only a configured limit judges them."""
    artifacts = _run_with_outcomes(tmp_path, landed=20, not_final=50, final=30)

    assert (artifacts["expected"], artifacts["missing"]) == (100, 0)
    assert (artifacts["failed"]["total"], artifacts["failed"]["not_final"], artifacts["failed"]["final"]) == (
        80,
        50,
        30,
    )
    assert artifacts["failed"]["by_class"] == {"agent_run_error": 80}
    assert artifacts["healthy"] is True
    assert artifacts["unhealthy_reasons"] == []


@pytest.mark.parametrize(
    ("outcomes", "reason"),
    [
        ({"landed": 99, "missing": 1}, "missing 1 of 100"),
        (
            {"landed": 98, "not_final": 2, "max_not_final_failure_fraction": 0.02},
            "not final failures 2 of 100 (2.0%), limit 2%",
        ),
        ({"landed": 97, "final": 3, "max_final_failure_fraction": 0.02}, "final failures 3 of 100 (3.0%), limit 2%"),
        (
            {"landed": 99, "not_final": 1, "max_not_final_failure_fraction": 0.005},
            "not final failures 1 of 100 (1.0%), limit 0.5%",
        ),
    ],
)
def test_run_is_unhealthy_with_one_reason_per_crossed_limit(tmp_path: Path, outcomes: dict, reason: str) -> None:
    artifacts = _run_with_outcomes(tmp_path, **outcomes)

    assert artifacts["healthy"] is False
    assert artifacts["unhealthy_reasons"] == [reason]


def test_summary_leaves_expected_missing_and_healthy_unknown_without_materialized_inputs(tmp_path: Path) -> None:
    rollout_path = _write_fixture(tmp_path, [(_record(0, 0), [_call()])])

    artifacts = run_health_checks(rollout_path, workers=1).summary["run"]["artifacts"]

    assert (artifacts["expected"], artifacts["missing"], artifacts["healthy"]) == (None, None, None)
    assert artifacts["unhealthy_reasons"] == []
    # No sidecar: nothing was measured, so not a count of 0.
    assert artifacts["failed"] is None


def test_failed_collection_still_writes_quality_summary_with_counts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_path = tmp_path / "rollouts.jsonl"
    (tmp_path / "rollouts_materialized_inputs.jsonl").write_bytes(
        b"".join(orjson.dumps({"_ng_task_index": task, "_ng_rollout_index": 0}) + b"\n" for task in range(3))
    )
    seen = {}

    async def failing_collection(self, config):
        seen["disable_health_check"] = config.disable_health_check
        _write_fixture(tmp_path, [(_record(0, 0), [_call()])])
        (tmp_path / "rollouts_failures.jsonl").write_bytes(
            orjson.dumps({"_ng_task_index": 1, "_ng_rollout_index": 0, "_ng_failure_class": "agent_run_error"}) + b"\n"
        )
        raise RuntimeError("EVAL FAILED: 1/3 samples completed.")

    monkeypatch.setattr(RolloutCollectionHelper, "run_from_config", failing_collection)
    monkeypatch.setattr(
        cli_eval,
        "get_global_config_dict",
        lambda: {"input_jsonl_fpath": str(tmp_path / "input.jsonl"), "output_jsonl_fpath": str(output_path)},
    )
    monkeypatch.setattr("nemo_gym.telemetry.setup.init_telemetry", lambda **kwargs: None)

    with pytest.raises(RuntimeError, match="EVAL FAILED"):
        cli_eval.collect_rollouts()

    artifacts = json.loads((tmp_path / "quality_summary.json").read_text())["run"]["artifacts"]
    assert seen["disable_health_check"] is True
    assert (artifacts["records"], artifacts["expected"], artifacts["missing"]) == (1, 3, 1)
    assert "partial" not in artifacts  # A Python error keeps the full check.
    assert artifacts["failed"] == {"total": 1, "final": 0, "not_final": 1, "by_class": {"agent_run_error": 1}}
    assert artifacts["healthy"] is False
    # No failure limit is configured, so only the missing rollout makes the run unhealthy.
    assert artifacts["unhealthy_reasons"] == ["missing 1 of 3"]


def _inventory(root: Path, tasks: int) -> None:
    (root / "rollouts_materialized_inputs.jsonl").write_bytes(
        b"".join(orjson.dumps({"_ng_task_index": task, "_ng_rollout_index": 0}) + b"\n" for task in range(tasks))
    )


def test_sigterm_during_collection_writes_a_provisional_summary_without_reading_rollouts(tmp_path: Path) -> None:
    """Slurm's scancel, time limit and preemption send SIGTERM; the run accounting is written within KillWait.

    The rollouts file is made unreadable before the signal, so the summary can only come from the keys collection
    holds in memory.
    """
    import signal
    import subprocess
    import time

    _inventory(tmp_path, 3)
    child = f"""
import asyncio
from pathlib import Path
import nemo_gym.cli.eval as cli_eval
from nemo_gym.rollout_collection import RolloutCollectionHelper

root = Path({str(tmp_path)!r})

async def collection(self, config):
    (root / "rollouts.jsonl").write_bytes((root / "landed.jsonl").read_bytes())
    self._landed_keys.add((0, 0))
    (root / "rollouts.jsonl").chmod(0)
    (root / "collecting").touch()
    await asyncio.sleep(600)

RolloutCollectionHelper.run_from_config = collection
cli_eval.get_global_config_dict = lambda: {{
    "input_jsonl_fpath": str(root / "input.jsonl"),
    "output_jsonl_fpath": str(root / "rollouts.jsonl"),
}}
import nemo_gym.telemetry.setup as telemetry
telemetry.init_telemetry = lambda **kwargs: None
cli_eval.collect_rollouts()
"""
    (tmp_path / "landed.jsonl").write_bytes(_write_fixture(tmp_path, [(_record(0, 0), [_call()])]).read_bytes())
    (tmp_path / "rollouts.jsonl").unlink()
    process = subprocess.Popen([sys.executable, "-c", child], cwd=Path(__file__).parents[2])
    try:
        deadline = time.monotonic() + 60
        while not (tmp_path / "collecting").exists():
            assert process.poll() is None and time.monotonic() < deadline, "collection never started"
            time.sleep(0.1)
        sent = time.monotonic()
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
        elapsed = time.monotonic() - sent
    finally:
        process.kill()

    summary = json.loads((tmp_path / "quality_summary.json").read_text())
    artifacts = summary["run"]["artifacts"]
    assert (artifacts["partial"], artifacts["expected"], artifacts["missing"]) == (True, 3, 2)
    assert artifacts["healthy"] is False
    assert artifacts["unhealthy_reasons"] == ["missing 2 of 3"]
    assert "verdicts" not in summary["run"]
    assert process.returncode == 143
    assert elapsed < 30


def test_a_new_run_removes_the_previous_jobs_health_reports(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("quality_summary.json", "rollout_verdicts.jsonl"):
        (tmp_path / name).write_text('{"run": {"artifacts": {"healthy": true}}}')
    seen = {}

    async def collection(self, config):
        seen["reports"] = sorted(
            path.name for path in tmp_path.iterdir() if path.name in ("quality_summary.json", "rollout_verdicts.jsonl")
        )

    monkeypatch.setattr(RolloutCollectionHelper, "run_from_config", collection)
    monkeypatch.setattr(
        cli_eval,
        "get_global_config_dict",
        lambda: {
            "input_jsonl_fpath": str(tmp_path / "input.jsonl"),
            "output_jsonl_fpath": str(tmp_path / "out.jsonl"),
        },
    )
    monkeypatch.setattr("nemo_gym.telemetry.setup.init_telemetry", lambda **kwargs: None)

    cli_eval.collect_rollouts()

    assert seen["reports"] == []


def test_duplicated_rollout_rows_make_the_run_unhealthy(tmp_path: Path) -> None:
    rollout_path = _write_fixture(tmp_path, [(_record(task, 0), [_call()]) for task in (0, 0, 1)])
    _inventory(tmp_path, 2)

    artifacts = run_health_checks(rollout_path, workers=1).summary["run"]["artifacts"]

    assert artifacts["healthy"] is False
    assert artifacts["unhealthy_reasons"] == ["2 records share a rollout identity"]


def test_a_limit_of_zero_with_no_failures_is_healthy(tmp_path: Path) -> None:
    artifacts = _run_with_outcomes(
        tmp_path, landed=100, max_not_final_failure_fraction=0, max_final_failure_fraction=0
    )

    assert artifacts["healthy"] is True


def test_only_inventory_rows_count_and_non_object_sidecar_lines_are_skipped(tmp_path: Path) -> None:
    rollout_path = _write_fixture(tmp_path, [(_record(task, 0), [_call()]) for task in range(3)])
    with rollout_path.open("ab") as rollouts:
        rollouts.write(b'{"_ng_task_index": 3')  # A killed writer's truncated record.
    _inventory(tmp_path, 4)
    (tmp_path / "rollouts_failures.jsonl").write_bytes(
        b'"x"\n[1]\n'
        + orjson.dumps({"_ng_task_index": 9, "_ng_rollout_index": 0, "_ng_failure_class": "agent_run_error"})
        + b"\n"
    )

    artifacts = run_health_checks(rollout_path, workers=1).summary["run"]["artifacts"]

    assert (artifacts["expected"], artifacts["missing"]) == (4, 1)
    assert artifacts["failed"]["total"] == 0


def test_the_reports_record_the_rollouts_they_were_computed_from(tmp_path: Path) -> None:
    """The reports keep one name per folder whatever the rollouts file is called, so the quality summary names the
    rollouts it was computed from, relative to its own folder (a Harbor run's summary links a report by it)."""
    rollout_path = _write_fixture(tmp_path, [(_record(0, 0), [_call()])])
    result = run_health_checks(rollout_path, workers=1)
    assert json.loads(result.summary_path.read_text())["run"]["rollouts"] == ["rollouts.jsonl"]
    elsewhere = run_health_checks(rollout_path, output_dir=tmp_path / "reports", workers=1)
    assert json.loads(elsewhere.summary_path.read_text())["run"]["rollouts"] == ["../rollouts.jsonl"]
    _inventory(tmp_path, 1)
    provisional = health.write_provisional_summary(rollout_path, {(0, 0)})
    assert json.loads(provisional.read_text())["run"]["rollouts"] == ["rollouts.jsonl"]


def test_a_provisional_summary_without_a_reason_leaves_healthy_unknown(tmp_path: Path) -> None:
    """Duplicates are found only by the per-record pass, so clean accounting does not prove the run healthy."""
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(b"")
    _inventory(tmp_path, 2)
    (tmp_path / "rollout_verdicts.jsonl").write_text("{}\n")

    summary_path = health.write_provisional_summary(rollout_path, {(0, 0), (1, 0)})

    artifacts = json.loads(summary_path.read_text())["run"]["artifacts"]
    assert (artifacts["partial"], artifacts["expected"], artifacts["missing"], artifacts["healthy"]) == (
        True,
        2,
        0,
        None,
    )
    assert not (tmp_path / "rollout_verdicts.jsonl").exists()


class TestOneSourceOfLimits:
    """The run-health limits come from one shared config, are persisted, and a recheck reuses them."""

    @staticmethod
    def _failed_run(root: Path) -> Path:
        # 100 tasks: 98 land, 2 fail for good. 2% is past a 1% limit for final failures.
        rollout_path = _write_fixture(root, [(_record(task, 0), [_call()]) for task in range(98)])
        _inventory(root, 100)
        (root / "rollouts_failures.jsonl").write_bytes(
            b"".join(
                orjson.dumps(
                    {
                        "_ng_task_index": task,
                        "_ng_rollout_index": 0,
                        "_ng_failure_class": "agent_run_error",
                        "_ng_failure_terminal": True,
                    }
                )
                + b"\n"
                for task in (98, 99)
            )
        )
        return rollout_path

    def test_the_effective_limits_are_written_to_the_summary(self, tmp_path: Path) -> None:
        from nemo_gym.health.types import RunHealthLimits

        result = run_health_checks(
            self._failed_run(tmp_path), workers=1, limits=RunHealthLimits(max_final_failure_fraction=0.01)
        )

        artifacts = json.loads(result.summary_path.read_text())["run"]["artifacts"]
        assert artifacts["limits"] == {
            "max_not_final_failure_fraction": None,
            "max_final_failure_fraction": 0.01,
            "source": {"max_not_final_failure_fraction": "config", "max_final_failure_fraction": "config"},
        }
        assert artifacts["healthy"] is False

    def test_a_recheck_uses_the_runs_persisted_limits(self, tmp_path: Path) -> None:
        from nemo_gym.health.types import RunHealthLimits

        rollout_path = self._failed_run(tmp_path)
        run_health_checks(rollout_path, workers=1, limits=RunHealthLimits(max_final_failure_fraction=0.01))

        recheck = health.health_check_run_dir(tmp_path, workers=1)

        artifacts = recheck.summary["run"]["artifacts"]
        assert artifacts["healthy"] is False  # The verdict does not flip on a recheck of the same artifacts.
        assert artifacts["limits"] == {
            "max_not_final_failure_fraction": None,
            "max_final_failure_fraction": 0.01,
            "source": {"max_not_final_failure_fraction": "persisted", "max_final_failure_fraction": "persisted"},
        }

    def test_an_explicit_override_wins_and_is_recorded(self, tmp_path: Path) -> None:
        from nemo_gym.health.types import RunHealthLimits

        rollout_path = self._failed_run(tmp_path)
        run_health_checks(rollout_path, workers=1, limits=RunHealthLimits(max_final_failure_fraction=0.01))

        recheck = health.health_check_run_dir(
            tmp_path, workers=1, limits=RunHealthLimits(max_final_failure_fraction=0.05)
        )

        artifacts = recheck.summary["run"]["artifacts"]
        assert artifacts["healthy"] is True
        assert artifacts["limits"]["max_final_failure_fraction"] == 0.05
        assert artifacts["limits"]["source"] == {
            "max_not_final_failure_fraction": "persisted",
            "max_final_failure_fraction": "override",
        }

    def test_the_provisional_summary_records_its_limits(self, tmp_path: Path) -> None:
        from nemo_gym.health.types import RunHealthLimits

        rollout_path = tmp_path / "rollouts.jsonl"
        rollout_path.write_bytes(b"")
        _inventory(tmp_path, 2)

        summary_path = health.write_provisional_summary(
            rollout_path, {(0, 0), (1, 0)}, limits=RunHealthLimits(max_not_final_failure_fraction=0.1)
        )

        limits = json.loads(summary_path.read_text())["run"]["artifacts"]["limits"]
        assert limits == {
            "max_not_final_failure_fraction": 0.1,
            "max_final_failure_fraction": None,
            "source": {"max_not_final_failure_fraction": "config", "max_final_failure_fraction": "config"},
        }

    @pytest.mark.parametrize("config_name", ["RolloutCollectionConfig", "RolloutAggregationConfig"])
    def test_every_command_reads_the_limits_from_the_shared_config(self, config_name: str) -> None:
        from nemo_gym.health.types import RunHealthConfig, RunHealthLimits

        config_cls = getattr(rollout_collection, config_name)
        assert issubclass(config_cls, RunHealthConfig)
        config = config_cls.model_validate(
            {
                "input_jsonl_fpath": "in.jsonl",
                "input_glob": "in.jsonl",
                "output_jsonl_fpath": "out.jsonl",
                "health_check_max_final_failure_fraction": 0.03,
            }
        )
        assert config.run_health_limits == RunHealthLimits(max_final_failure_fraction=0.03)

    def test_health_check_cli_passes_the_limit_overrides(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nemo_gym.health.types import RunHealthLimits

        received = {}
        monkeypatch.setattr(cli_eval, "health_check_rollouts", lambda run_dir, **kwargs: received.update(kwargs))
        monkeypatch.setattr(
            sys,
            "argv",
            ["gym", "eval", "health-check", str(tmp_path), "--health-max-final-failure-fraction", "0.05"],
        )

        cli_main.main()

        assert received["limits"] == RunHealthLimits(max_final_failure_fraction=0.05)


class TestPerFlagLimitOverride:
    """``gym eval health-check``: each limit flag overrides only its own limit; the other keeps the persisted one."""

    @staticmethod
    def _judged_run(root: Path) -> None:
        # 100 tasks: 96 land, 2 fail for good, 2 fail but are retryable. Both 2% shares reach a 1% limit.
        _write_fixture(root, [(_record(task, 0), [_call()]) for task in range(96)])
        _inventory(root, 100)
        (root / "rollouts_failures.jsonl").write_bytes(
            b"".join(
                orjson.dumps(
                    {
                        "_ng_task_index": task,
                        "_ng_rollout_index": 0,
                        "_ng_failure_class": "agent_run_error",
                        "_ng_failure_terminal": task in (96, 97),
                    }
                )
                + b"\n"
                for task in (96, 97, 98, 99)
            )
        )
        from nemo_gym.health.types import RunHealthLimits

        run_health_checks(
            root / "rollouts.jsonl",
            workers=1,
            limits=RunHealthLimits(max_not_final_failure_fraction=0.01, max_final_failure_fraction=0.01),
        )

    @staticmethod
    def _recheck(root: Path, monkeypatch: pytest.MonkeyPatch, flags: list[str]) -> dict:
        monkeypatch.setattr(sys, "argv", ["gym", "eval", "health-check", str(root), "--workers", "1", *flags])
        cli_main.main()
        return json.loads((root / "quality_summary.json").read_text())["run"]["artifacts"]

    @pytest.mark.parametrize(
        ("flags", "limits", "unhealthy_groups"),
        [
            pytest.param(
                ["--health-max-final-failure-fraction", "0.05"],
                {
                    "max_not_final_failure_fraction": 0.01,
                    "max_final_failure_fraction": 0.05,
                    "source": {
                        "max_not_final_failure_fraction": "persisted",
                        "max_final_failure_fraction": "override",
                    },
                },
                ["not final"],
                id="only-final-flag",
            ),
            pytest.param(
                ["--health-max-not-final-failure-fraction", "0.05"],
                {
                    "max_not_final_failure_fraction": 0.05,
                    "max_final_failure_fraction": 0.01,
                    "source": {
                        "max_not_final_failure_fraction": "override",
                        "max_final_failure_fraction": "persisted",
                    },
                },
                ["final"],
                id="only-not-final-flag",
            ),
            pytest.param(
                ["--health-max-not-final-failure-fraction", "0.05", "--health-max-final-failure-fraction", "0.05"],
                {
                    "max_not_final_failure_fraction": 0.05,
                    "max_final_failure_fraction": 0.05,
                    "source": {"max_not_final_failure_fraction": "override", "max_final_failure_fraction": "override"},
                },
                [],
                id="both-flags",
            ),
            pytest.param(
                [],
                {
                    "max_not_final_failure_fraction": 0.01,
                    "max_final_failure_fraction": 0.01,
                    "source": {
                        "max_not_final_failure_fraction": "persisted",
                        "max_final_failure_fraction": "persisted",
                    },
                },
                ["not final", "final"],
                id="neither-flag",
            ),
        ],
    )
    def test_each_flag_overrides_only_its_own_limit(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        flags: list[str],
        limits: dict,
        unhealthy_groups: list[str],
    ) -> None:
        self._judged_run(tmp_path)

        artifacts = self._recheck(tmp_path, monkeypatch, flags)

        assert artifacts["limits"] == limits
        assert [reason.split(" failures")[0] for reason in artifacts["unhealthy_reasons"]] == unhealthy_groups
        assert artifacts["healthy"] is (not unhealthy_groups)

    def test_quality_summary_records_the_source_of_each_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._judged_run(tmp_path)
        first = json.loads((tmp_path / "quality_summary.json").read_text())["run"]["artifacts"]["limits"]
        assert first["source"] == {"max_not_final_failure_fraction": "config", "max_final_failure_fraction": "config"}

        overridden = self._recheck(tmp_path, monkeypatch, ["--health-max-final-failure-fraction", "0.05"])["limits"]
        assert overridden["source"] == {
            "max_not_final_failure_fraction": "persisted",
            "max_final_failure_fraction": "override",
        }

        # The override is persisted in turn, so a recheck without flags reproduces the overridden verdict.
        again = self._recheck(tmp_path, monkeypatch, [])["limits"]
        assert again == {
            "max_not_final_failure_fraction": 0.01,
            "max_final_failure_fraction": 0.05,
            "source": {"max_not_final_failure_fraction": "persisted", "max_final_failure_fraction": "persisted"},
        }


class TestNoLimitValue:
    """``none`` on a limit flag removes that one limit; a flag that is not given still keeps the persisted one."""

    @pytest.mark.parametrize(
        ("flag", "removed", "kept", "remaining_group"),
        [
            pytest.param(
                "--health-max-final-failure-fraction",
                "max_final_failure_fraction",
                "max_not_final_failure_fraction",
                "not final",
                id="final",
            ),
            pytest.param(
                "--health-max-not-final-failure-fraction",
                "max_not_final_failure_fraction",
                "max_final_failure_fraction",
                "final",
                id="not-final",
            ),
        ],
    )
    def test_none_removes_only_that_limit_and_a_recheck_keeps_it_removed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str, removed: str, kept: str, remaining_group: str
    ) -> None:
        TestPerFlagLimitOverride._judged_run(tmp_path)

        artifacts = TestPerFlagLimitOverride._recheck(tmp_path, monkeypatch, [flag, "none"])

        assert artifacts["limits"] == {
            "max_not_final_failure_fraction": None if removed == "max_not_final_failure_fraction" else 0.01,
            "max_final_failure_fraction": None if removed == "max_final_failure_fraction" else 0.01,
            "source": {removed: "override", kept: "persisted"},
        }
        # Both groups were over their limit; only the one that keeps a limit is still a reason.
        assert [reason.split(" failures")[0] for reason in artifacts["unhealthy_reasons"]] == [remaining_group]

        again = TestPerFlagLimitOverride._recheck(tmp_path, monkeypatch, [])

        assert again["limits"][removed] is None
        assert again["limits"]["source"] == {removed: "persisted", kept: "persisted"}
        assert [reason.split(" failures")[0] for reason in again["unhealthy_reasons"]] == [remaining_group]

    def test_none_on_both_flags_makes_the_run_healthy(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        TestPerFlagLimitOverride._judged_run(tmp_path)

        artifacts = TestPerFlagLimitOverride._recheck(
            tmp_path,
            monkeypatch,
            ["--health-max-final-failure-fraction", "none", "--health-max-not-final-failure-fraction", "none"],
        )

        assert artifacts["healthy"] is True
        assert artifacts["limits"]["max_final_failure_fraction"] is None
        assert artifacts["limits"]["max_not_final_failure_fraction"] is None

    @pytest.mark.parametrize("value", ["none", "None", "NONE", "nOnE"])
    def test_none_is_accepted_in_any_case(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str) -> None:
        from nemo_gym.health.types import RunHealthLimits

        received = {}
        monkeypatch.setattr(cli_eval, "health_check_rollouts", lambda run_dir, **kwargs: received.update(kwargs))
        monkeypatch.setattr(
            sys, "argv", ["gym", "eval", "health-check", str(tmp_path), "--health-max-final-failure-fraction", value]
        )

        cli_main.main()

        limits = received["limits"]
        assert limits == RunHealthLimits(max_final_failure_fraction=None)
        assert limits.model_fields_set == {"max_final_failure_fraction"}

    def test_a_flag_that_is_not_given_leaves_its_limit_unset(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        received = {}
        monkeypatch.setattr(cli_eval, "health_check_rollouts", lambda run_dir, **kwargs: received.update(kwargs))
        monkeypatch.setattr(
            sys,
            "argv",
            ["gym", "eval", "health-check", str(tmp_path), "--health-max-not-final-failure-fraction", "0.2"],
        )

        cli_main.main()

        assert received["limits"].model_fields_set == {"max_not_final_failure_fraction"}
        assert received["limits"].max_not_final_failure_fraction == 0.2

    @pytest.mark.parametrize("value", ["nil", "", "1.5", "-0.1", "nan", "inf"])
    def test_an_invalid_value_is_an_argparse_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        value: str,
    ) -> None:
        monkeypatch.setattr(cli_eval, "health_check_rollouts", lambda run_dir, **kwargs: pytest.fail("must not run"))
        monkeypatch.setattr(
            sys, "argv", ["gym", "eval", "health-check", str(tmp_path), "--health-max-final-failure-fraction", value]
        )

        with pytest.raises(SystemExit) as exc:
            cli_main.main()

        assert exc.value.code == 2
        assert "--health-max-final-failure-fraction" in capsys.readouterr().err

    def test_python_api_explicit_none_removes_and_unset_keeps(self, tmp_path: Path) -> None:
        from nemo_gym.health.types import RunHealthLimits

        TestPerFlagLimitOverride._judged_run(tmp_path)

        result = health.health_check_run_dir(
            tmp_path, workers=1, limits=RunHealthLimits(max_final_failure_fraction=None)
        )

        limits = result.summary["run"]["artifacts"]["limits"]
        assert limits == {
            "max_not_final_failure_fraction": 0.01,
            "max_final_failure_fraction": None,
            "source": {"max_not_final_failure_fraction": "persisted", "max_final_failure_fraction": "override"},
        }

        unchanged = health.health_check_run_dir(tmp_path, workers=1, limits=RunHealthLimits())

        assert unchanged.summary["run"]["artifacts"]["limits"] == {
            "max_not_final_failure_fraction": 0.01,
            "max_final_failure_fraction": None,
            "source": {"max_not_final_failure_fraction": "persisted", "max_final_failure_fraction": "persisted"},
        }


class TestRunExcessiveFailuresCheck:
    """The run-level failure limits are one check of the suite: findings, coverage, and an ignore switch."""

    @staticmethod
    def _summary(root: Path, *, ignored: tuple[str, ...] = (), **limits: float) -> dict:
        from nemo_gym.health.types import RunHealthLimits

        rollout_path = TestOneSourceOfLimits._failed_run(root)
        result = run_health_checks(rollout_path, workers=1, ignored_checks=ignored, limits=RunHealthLimits(**limits))
        return json.loads(result.summary_path.read_text())["run"]

    def test_the_check_is_registered_at_run_scope(self) -> None:
        from nemo_gym.health.checks import normalize_ignored_checks
        from nemo_gym.health.types import CheckScope

        [spec] = [spec for spec in CHECK_REGISTRY if spec.id == "run_excessive_failures"]
        assert spec.evaluation_scope == CheckScope.RUN
        assert normalize_ignored_checks("run_excessive_failures") == ("run_excessive_failures",)

    def test_a_crossed_limit_is_a_finding_with_coverage(self, tmp_path: Path) -> None:
        run = self._summary(tmp_path, max_final_failure_fraction=0.01)

        assert run["issues"]["run_excessive_failures"] == 1
        assert run["artifacts"]["coverage"]["run_excessive_failures"] == {
            "evaluated": 1,
            "unobserved": 0,
            "ignored": 0,
        }
        [finding] = run["findings"]
        assert finding["check"] == "run_excessive_failures"
        assert finding["detail"] == {"group": "final", "count": 2, "expected": 100, "limit": 0.01}
        assert run["artifacts"]["healthy"] is False

    def test_an_ignored_check_judges_nothing_but_the_counts_stay(self, tmp_path: Path) -> None:
        run = self._summary(tmp_path, ignored=("run_excessive_failures",), max_final_failure_fraction=0.01)

        assert run["artifacts"]["coverage"]["run_excessive_failures"] == {
            "evaluated": 0,
            "unobserved": 0,
            "ignored": 1,
        }
        assert run["issues"]["run_excessive_failures"] == 0
        assert run["findings"] == []
        assert run["artifacts"]["healthy"] is True
        assert run["artifacts"]["failed"]["final"] == 2
        assert "run_excessive_failures" in run["ignored_checks"]

    def test_without_a_limit_the_check_is_not_evaluated(self, tmp_path: Path) -> None:
        run = self._summary(tmp_path)

        assert run["artifacts"]["coverage"]["run_excessive_failures"] == {
            "evaluated": 0,
            "unobserved": 1,
            "ignored": 0,
        }
        assert run["issues"]["run_excessive_failures"] == 0
        assert run["artifacts"]["healthy"] is True

    @pytest.mark.parametrize("ignored", [(), ("run_excessive_failures",)], ids=["evaluated", "ignored"])
    def test_the_provisional_summary_evaluates_only_this_check(self, tmp_path: Path, ignored: tuple) -> None:
        from nemo_gym.health.types import RunHealthLimits

        rollout_path = TestOneSourceOfLimits._failed_run(tmp_path)

        summary_path = health.write_provisional_summary(
            rollout_path,
            {(task, 0) for task in range(98)},
            limits=RunHealthLimits(max_final_failure_fraction=0.01),
            ignored_checks=ignored,
        )

        run = json.loads(summary_path.read_text())["run"]
        others = sorted(spec.id for spec in CHECK_REGISTRY if spec.id != "run_excessive_failures")
        assert run["artifacts"]["not_evaluated"] == others
        coverage = run["artifacts"]["coverage"]
        if ignored:
            assert coverage == {"run_excessive_failures": {"evaluated": 0, "unobserved": 0, "ignored": 1}}
            assert run["findings"] == []
            assert run["artifacts"]["healthy"] is None  # The accounting alone proves nothing here.
        else:
            assert coverage == {"run_excessive_failures": {"evaluated": 1, "unobserved": 0, "ignored": 0}}
            assert [finding["check"] for finding in run["findings"]] == ["run_excessive_failures"]
            assert run["artifacts"]["healthy"] is False


class TestFailureGroupsAreNamedByRetry:
    """The two failure groups are split by whether a failure is final; the cause is not known, so not claimed."""

    def test_the_limits_name_final_and_not_final_failures(self) -> None:
        from nemo_gym.health.types import RunHealthConfig, RunHealthLimits

        assert set(RunHealthLimits.model_fields) == {"max_final_failure_fraction", "max_not_final_failure_fraction"}
        assert {name for name in RunHealthConfig.model_fields if "failure_fraction" in name} == {
            "health_check_max_final_failure_fraction",
            "health_check_max_not_final_failure_fraction",
        }
        descriptions = " ".join(field.description or "" for field in RunHealthConfig.model_fields.values())
        assert "infra" not in descriptions and "model" not in descriptions

    def test_a_not_final_finding_and_reason_claim_no_cause(self, tmp_path: Path) -> None:
        from nemo_gym.health.types import RunHealthLimits

        rollout_path = _write_fixture(tmp_path, [(_record(task, 0), [_call()]) for task in range(98)])
        _inventory(tmp_path, 100)
        (tmp_path / "rollouts_failures.jsonl").write_bytes(
            b"".join(
                orjson.dumps({"_ng_task_index": task, "_ng_rollout_index": 0, "_ng_failure_class": "judge_failed"})
                + b"\n"
                for task in (98, 99)
            )
        )

        result = run_health_checks(
            rollout_path, workers=1, limits=RunHealthLimits(max_not_final_failure_fraction=0.01)
        )

        run = json.loads(result.summary_path.read_text())["run"]
        [finding] = run["findings"]
        assert finding["detail"]["group"] == "not_final"
        assert run["artifacts"]["unhealthy_reasons"] == ["not final failures 2 of 100 (2.0%), limit 1%"]
        assert "infra" not in json.dumps(run["artifacts"]["limits"])

    @pytest.mark.parametrize("command", ["run", "aggregate"])
    def test_the_cli_flag_names_the_not_final_group(self, command: str, monkeypatch: pytest.MonkeyPatch) -> None:
        from tests.unit_tests.test_cli_main import _dispatch_for

        _, overrides = _dispatch_for(monkeypatch, ["eval", command, "--health-max-not-final-failure-fraction", "0.05"])
        assert overrides == ["+health_check_max_not_final_failure_fraction=0.05"]
