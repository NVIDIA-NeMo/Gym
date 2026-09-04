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
from nemo_gym.base_responses_api_model import build_model_call_record
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper, _CompletedRollout
from nemo_gym.rollout_health import CHECK_REGISTRY, run_health_checks
from nemo_gym.rollout_observability import TrajectoryRecord


MODEL_CALL_CHECKS = {
    "model_call_zero_completion_tokens",
    "model_call_missing_token_counts",
    "trajectory_capture_mismatch",
    "model_call_failed",
    "rollout_token_count_mismatch",
    "model_call_runaway_generation",
}
RUNNER_CHECKS = {"check_execution_error", "record_unreadable"}


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
        "request": call.get("request"),
        "response": call.get("response"),
        "response_metadata": {
            "response_id": call.get("response_id"),
            "model_ref": call.get("model_ref"),
            "status_code": call.get("status_code"),
            "response_status": call.get("response_status"),
            "finish_reason": call.get("finish_reason"),
            "error_category": call.get("error_category"),
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
    assert by_id["model_call_failed"].evaluation_scope == health.CheckScope.ROLLOUT
    assert by_id["model_call_failed"].subject == health.CheckSubject.MODEL_CALL
    assert by_id["task_consistently_unhealthy"].evaluation_scope == health.CheckScope.TASK
    assert by_id["task_consistently_unhealthy"].subject == health.CheckSubject.TASK
    assert by_id["trajectory_capture_mismatch"].reads == frozenset(
        {health.CheckInput.RECORD, health.CheckInput.TRAJECTORY, health.CheckInput.BOUND_CALLS}
    )
    assert by_id["agent_turn_hollow"].reads == frozenset(
        {health.CheckInput.RECORD, health.CheckInput.TRAJECTORY, health.CheckInput.AGENT_TURNS}
    )
    assert by_id["rollout_token_count_mismatch"].reads == frozenset(
        {health.CheckInput.RECORD, health.CheckInput.TRAJECTORY, health.CheckInput.BOUND_CALLS}
    )


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
    assert all(result.summary["run"]["issues"][spec.id] > 0 for spec in CHECK_REGISTRY if spec.id not in RUNNER_CHECKS)
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
    assert set(digest.unobserved) == MODEL_CALL_CHECKS
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

        async def _call_aggregate_metrics(self, results, rows, output_fpath):
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

    def fake_health_check(run_dir, *, rollout_file=None, workers=None, ignored_checks=(), json_output=False):
        received.update(
            run_dir=run_dir,
            rollout_file=rollout_file,
            workers=workers,
            ignored_checks=ignored_checks,
            json_output=json_output,
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
    assert set(digest.unobserved) == MODEL_CALL_CHECKS
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
    assert not any(finding.check == "model_call_failed" for finding in result.rollouts[0].findings)
    assert {
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "model_call_failed",
        "rollout_token_count_mismatch",
        "model_call_runaway_generation",
    } <= set(result.rollouts[0].unobserved)
    assert result.summary["run"]["stats"]["duplicated_calls"] == {"replayed": 0, "rollouts": 0}
    assert health_checks._call_identity({"response_id": "loose"}) == "response::loose"
    assert health_checks._call_identity({}) is None


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
    assert checks.count("model_call_failed") == 1
    assert checks.count("rollout_token_count_mismatch") == 1
    assert "trajectory_capture_mismatch" not in checks
    failed = next(finding for finding in result.rollouts[0].findings if finding.check == "model_call_failed")
    assert failed.detail == {"status": 500, "error_category": "upstream", "terminal": True}


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
    assert health_checks._response_has_content("malformed") is False
    assert health_checks._response_has_content({"content": "visible"}) is True


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
        "agent_turn_hollow",
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "trajectory_capture_mismatch",
        "model_call_failed",
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
