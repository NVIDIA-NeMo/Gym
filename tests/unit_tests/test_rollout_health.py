# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
import warnings
from asyncio import Future
from pathlib import Path

import orjson
import pytest

import nemo_gym.cli.eval as cli_eval
import nemo_gym.cli.main as cli_main
import nemo_gym.health.checks as health_checks
import nemo_gym.rollout_collection as rollout_collection
import nemo_gym.rollout_health as health
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollout_health import CHECK_REGISTRY, run_health_checks
from nemo_gym.rollout_observability import TrajectoryRecord


CAPTURE_CHECKS = {
    "model_call_zero_completion_tokens",
    "model_call_missing_token_counts",
    "trajectory_capture_mismatch",
    "model_call_failed",
    "rollout_token_count_mismatch",
    "model_call_runaway_generation",
}


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


def _write_fixture(root: Path, rows: list[tuple[dict, list[dict]]]) -> tuple[Path, Path]:
    rollout_path = root / "rollouts.jsonl"
    capture_dir = root / "captures"
    capture_dir.mkdir(parents=True)
    with rollout_path.open("wb") as rollouts:
        for record, calls in rows:
            rollouts.write(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
            rollout_id = f"{record['_ng_task_index']}-{record['_ng_rollout_index']}"
            with (capture_dir / f"{rollout_id}.capture.jsonl").open("wb") as capture:
                for call in calls:
                    capture.write(orjson.dumps(call, option=orjson.OPT_APPEND_NEWLINE))
    return rollout_path, capture_dir


def test_check_ids_encode_subject_without_replacing_evaluation_scope() -> None:
    assert all(spec.id.startswith(f"{spec.subject.value}_") for spec in CHECK_REGISTRY)
    by_id = {spec.id: spec for spec in CHECK_REGISTRY}
    assert by_id["model_call_failed"].evaluation_scope == health.CheckScope.ROLLOUT
    assert by_id["model_call_failed"].subject == health.CheckSubject.MODEL_CALL
    assert by_id["task_consistently_unhealthy"].evaluation_scope == health.CheckScope.TASK
    assert by_id["task_consistently_unhealthy"].subject == health.CheckSubject.TASK
    assert by_id["trajectory_capture_mismatch"].reads == frozenset(
        {health.CheckInput.RECORD, health.CheckInput.CAPTURE}
    )
    assert by_id["rollout_token_count_mismatch"].reads == frozenset(
        {health.CheckInput.RECORD, health.CheckInput.CAPTURE, health.CheckInput.BOUND_CALLS}
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
    rollout_path, capture_dir = _write_fixture(tmp_path, rows)

    result = run_health_checks(rollout_path, capture_dirs=[capture_dir], capture_enabled=True, workers=2)

    assert set(result.summary["run"]["issues"]) == {spec.id for spec in CHECK_REGISTRY}
    assert all(
        result.summary["run"]["issues"][spec.id] > 0 for spec in CHECK_REGISTRY if spec.id != "record_unreadable"
    )
    assert result.summary["run"]["issues"]["record_unreadable"] == 0
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


@pytest.mark.parametrize(
    "state",
    ["capture off", "uncorrelated", "driver bypass"],
)
def test_each_capture_unobserved_state_is_not_unhealthy(tmp_path: Path, state: str) -> None:
    run_dir = tmp_path / state.replace(" ", "-")
    run_dir.mkdir()
    rollout_path = run_dir / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(_record(0, 0), option=orjson.OPT_APPEND_NEWLINE))
    capture_dir = run_dir / "captures"
    capture_dirs: list[Path] = []
    capture_enabled: bool | None = False if state == "capture off" else True
    driver_bypass = state == "driver bypass"
    if driver_bypass:
        capture_dir.mkdir()
        (capture_dir / "0-0.capture.jsonl").write_bytes(orjson.dumps(_call(), option=orjson.OPT_APPEND_NEWLINE))
        capture_dirs = [capture_dir]

    result = run_health_checks(
        rollout_path,
        capture_dirs=capture_dirs,
        capture_enabled=capture_enabled,
        driver_bypass=driver_bypass,
        workers=1,
    )

    [digest] = result.rollouts
    assert digest.verdict == "unobserved"
    assert set(digest.unobserved) == CAPTURE_CHECKS
    assert not digest.findings
    assert result.summary["run"]["verdicts"] == {"healthy": 0, "unhealthy": 0, "unobserved": 1}


async def test_health_on_and_off_leave_collection_and_metrics_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(rollout_collection, "get_global_config_dict", lambda: {})
    source = {
        "responses_create_params": {"input": []},
        "agent_ref": {"name": "synthetic-agent"},
    }

    class GoldenHelper(RolloutCollectionHelper):
        def run_examples(self, examples, *args, **kwargs):
            futures = []
            for example in examples:
                future = Future()
                future.set_result(
                    (
                        example,
                        {
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

    assert artifacts[False] == artifacts[True]


def test_health_check_cli_accepts_run_dir_workers_and_ignored_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    received = {}

    def fake_health_check(run_dir, *, workers=None, ignored_checks=()):
        received.update(run_dir=run_dir, workers=workers, ignored_checks=ignored_checks)

    monkeypatch.setattr(cli_eval, "health_check_rollouts", fake_health_check)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "gym",
            "eval",
            "health-check",
            str(tmp_path),
            "--workers",
            "3",
            "--ignore-checks",
            "model_call_missing_token_counts,model_call_zero_completion_tokens",
        ],
    )

    cli_main.main()

    assert received == {
        "run_dir": str(tmp_path),
        "workers": 3,
        "ignored_checks": ["model_call_missing_token_counts", "model_call_zero_completion_tokens"],
    }


def test_invocation_fallback_warns_and_remains_a_capture_input(tmp_path: Path) -> None:
    record = {
        "_ng_task_index": "task-a",
        "_ng_rollout_index": "repeat-a",
        "response": {"usage": {"prompt_tokens": 4, "completion_tokens": 2}},
        "ng_trajectory": {
            "rollout_id": "explicit-rollout",
            "turns": ["malformed-turn"],
            "invocations": [
                "malformed-invocation",
                {
                    "invocation_id": "root",
                    "model_calls": [
                        {
                            "model_ref": {"type": "responses_api_models", "name": "model"},
                            "response_id": "response-1",
                        },
                        {"response_id": "unqualified-response"},
                        None,
                        {},
                    ],
                    "conversation": [
                        "malformed-item",
                        {"role": "user", "content": "question"},
                        {"type": "function_call", "name": "tool", "arguments": {}},
                    ],
                },
            ],
            "model_calls": [
                "malformed-call",
                {
                    "model_call_id": None,
                    "request": {"input": "question"},
                    "response": {"content": "answer"},
                    "response_metadata": {
                        "response_id": "response-1",
                        "model_ref": {"type": "responses_api_models", "name": "model"},
                        "status_code": 200,
                        "response_status": "completed",
                    },
                    "token_stats": {"prompt_tokens": 4, "completion_tokens": 2},
                },
            ],
        },
    }
    rollout_path = tmp_path / "stored.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    with pytest.warns(RuntimeWarning, match="coarse ng_trajectory.invocations evidence"):
        result = run_health_checks(rollout_path, workers=1)

    [digest] = result.rollouts
    assert digest.rollout_id == "explicit-rollout"
    assert digest.capture_observed
    assert digest.model_calls == 1
    assert digest.capture_prompt_tokens == 4
    assert not any(finding.check == "rollout_missing_agent_turns" for finding in digest.findings)
    assert health_checks._normalized_embedded_calls(
        {"ng_model_call_capture": {"calls": [None, {"call_index": 1}]}}
    ) == [{"call_index": 1}]
    assert health_checks._nonempty(123) is False
    assert health_checks._call_ref_key({"response_id": "unqualified-response"}) is None
    assert health_checks._item_has_tool_call("bad") is False
    assert health_checks._item_is_agent_content("bad") is False


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

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = run_health_checks(rollout_path, capture_enabled=False, workers=1)

    steps, source = health_checks._agent_steps_with_source(record)
    assert source == "trajectory_turns"
    assert len(steps) == 1
    assert steps[0].has_message and steps[0].has_tool_calls
    assert not any(finding.check == "agent_turn_hollow" for finding in result.rollouts[0].findings)
    assert not any("ng_trajectory.turns was unavailable" in str(warning.message) for warning in caught)


def test_current_invocation_reasoning_is_message_content(tmp_path: Path) -> None:
    trajectory = TrajectoryRecord.model_validate(
        {
            "task_id": "0",
            "rollout_id": "0-0",
            "invocations": [
                {
                    "invocation_id": "root",
                    "model_calls": [{"model_call_id": "call-1"}],
                    "conversation": [
                        {
                            "type": "reasoning",
                            "id": "reasoning-1",
                            "summary": [{"type": "summary_text", "text": "thinking"}],
                        }
                    ],
                }
            ],
        }
    )
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "ng_trajectory": trajectory.model_dump(mode="json"),
    }
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    with pytest.warns(RuntimeWarning, match="coarse ng_trajectory.invocations evidence"):
        result = run_health_checks(rollout_path, capture_enabled=False, workers=1)

    [step] = health_checks._agent_steps(record)
    assert step.has_message and not step.has_tool_calls
    assert not any(finding.check == "agent_turn_hollow" for finding in result.rollouts[0].findings)


def test_noncanonical_transcript_warning_is_aggregated_and_only_emitted_when_used(tmp_path: Path) -> None:
    invocation_record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "ng_trajectory": {
            "task_id": "0",
            "rollout_id": "0-0",
            "invocations": [
                {
                    "invocation_id": "root",
                    "conversation": [{"role": "assistant", "content": "answer"}],
                }
            ],
        },
    }
    response_record = {
        "_ng_task_index": 1,
        "_ng_rollout_index": 0,
        "response": {"output": [{"role": "assistant", "content": "answer"}]},
    }
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(
        orjson.dumps(invocation_record, option=orjson.OPT_APPEND_NEWLINE)
        + orjson.dumps(response_record, option=orjson.OPT_APPEND_NEWLINE)
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_health_checks(rollout_path, capture_enabled=False, workers=1)

    fallback_warnings = [
        warning for warning in caught if "ng_trajectory.turns was unavailable" in str(warning.message)
    ]
    assert len(fallback_warnings) == 1
    message = str(fallback_warnings[0].message)
    assert "1 used coarse ng_trajectory.invocations evidence" in message
    assert "1 used heuristic response.output grouping" in message

    with warnings.catch_warnings(record=True) as ignored_caught:
        warnings.simplefilter("always")
        run_health_checks(
            rollout_path,
            capture_enabled=False,
            ignored_checks=sorted(health_checks._FALLBACK_TRANSCRIPT_CHECK_IDS),
            workers=1,
        )
    assert not any("ng_trajectory.turns was unavailable" in str(warning.message) for warning in ignored_caught)


@pytest.mark.parametrize("item_type", sorted(health_checks._AGENT_TOOL_CALL_TYPES))
def test_current_response_tool_call_types_count_as_agent_activity(item_type: str) -> None:
    item = {"type": item_type}
    assert health_checks._item_has_tool_call(item)
    assert health_checks._item_is_agent_content(item)


@pytest.mark.parametrize("item_type", sorted(health_checks._AGENT_TURN_BOUNDARY_TYPES))
def test_current_response_tool_result_types_end_agent_turn(item_type: str) -> None:
    assert health_checks._item_ends_agent_turn({"type": item_type})


def test_current_response_refusal_counts_as_message_content() -> None:
    assert health_checks._nonempty({"type": "refusal", "refusal": "cannot comply"})


def test_response_output_groups_agent_items_into_turns_and_keeps_reasoning(tmp_path: Path) -> None:
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "response": {
            "output": [
                {"type": "reasoning", "summary": [{"type": "summary_text", "text": "thinking"}]},
                {"type": "message", "role": "assistant", "content": "\n"},
                {"type": "function_call", "name": "tool", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "call-1", "output": "done"},
                {"type": "message", "role": "assistant", "content": "finished"},
                {"type": "message", "role": "user", "content": "ignored boundary"},
            ]
        },
    }
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    with pytest.warns(RuntimeWarning, match="heuristic response.output grouping"):
        result = run_health_checks(rollout_path, capture_enabled=False, workers=1)

    steps = health_checks._agent_steps(record)
    assert len(steps) == 2
    assert steps[0].has_message and steps[0].has_tool_calls
    assert steps[1].has_message and not steps[1].has_tool_calls
    assert not any(finding.check == "agent_turn_hollow" for finding in result.rollouts[0].findings)


def test_missing_all_bindings_is_unobserved_and_embedded_capture_is_used(tmp_path: Path) -> None:
    record = {
        "_ng_task_index": 0,
        "_ng_rollout_index": 0,
        "response": {
            "output": [
                {"type": "message", "role": "assistant", "content": "\n"},
                {"type": "function_call", "name": "tool", "arguments": "{}"},
            ]
        },
        "ng_model_call_capture": {"calls": [_call()]},
    }
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
    unrelated_capture_dir = tmp_path / "model_calls"
    unrelated_capture_dir.mkdir()
    (unrelated_capture_dir / "another-rollout.capture.jsonl").write_bytes(
        orjson.dumps(_call(), option=orjson.OPT_APPEND_NEWLINE)
    )

    result = run_health_checks(
        rollout_path,
        capture_dirs=[unrelated_capture_dir],
        capture_enabled=True,
        workers=1,
    )

    [digest] = result.rollouts
    assert digest.capture_observed
    assert digest.model_calls == 1
    assert digest.capture_prompt_tokens == 3
    assert set(digest.unobserved) == CAPTURE_CHECKS
    assert digest.verdict == "unobserved"
    assert not digest.findings


def test_ignored_check_is_excluded_from_execution_and_verdict(tmp_path: Path) -> None:
    rollout_path, capture_dir = _write_fixture(
        tmp_path,
        [(_record(0, 0, usage={"input_tokens": 3, "output_tokens": 0}), [_call(tokens_out=0)])],
    )

    result = run_health_checks(
        rollout_path,
        capture_dirs=[capture_dir],
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
    rollout_path, capture_dir = _write_fixture(
        tmp_path,
        [
            (_record(0, 0, usage={"input_tokens": 3, "output_tokens": 0}), [_call(tokens_out=0)]),
            (_record(0, 1, usage={"input_tokens": 3, "output_tokens": 0}), [_call(tokens_out=0)]),
        ],
    )

    result = run_health_checks(
        rollout_path,
        capture_dirs=[capture_dir],
        capture_enabled=True,
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


def test_empty_embedded_capture_is_unobserved(tmp_path: Path) -> None:
    record = _record(0, 0)
    record["ng_model_call_capture"] = {"calls": []}
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))

    result = run_health_checks(rollout_path, capture_enabled=True, workers=1)

    [digest] = result.rollouts
    assert not digest.capture_observed
    assert digest.model_calls == 0
    assert set(digest.unobserved) == CAPTURE_CHECKS
    assert digest.verdict == "unobserved"


def test_turn_without_a_model_call_reference_is_not_a_token_count_failure(tmp_path: Path) -> None:
    record = _record(0, 0, usage={"input_tokens": 3, "output_tokens": 2})
    record["ng_trajectory"]["turns"].append(
        {
            "turn_no": 2,
            "answer": "second answer",
            "model_calls": [],
        }
    )
    rollout_path, capture_dir = _write_fixture(tmp_path, [(record, [_call()])])

    result = run_health_checks(rollout_path, capture_dirs=[capture_dir], capture_enabled=True, workers=1)

    [digest] = result.rollouts
    assert digest.verdict == "healthy"
    assert digest.unobserved == []
    assert not any(finding.check == "model_call_missing_token_counts" for finding in digest.findings)


def test_bound_call_without_token_counts_is_a_model_call_finding(tmp_path: Path) -> None:
    rollout_path, capture_dir = _write_fixture(tmp_path, [(_record(0, 0), [_call(tokens_out=None)])])

    result = run_health_checks(rollout_path, capture_dirs=[capture_dir], capture_enabled=True, workers=1)

    findings = [
        finding for finding in result.rollouts[0].findings if finding.check == "model_call_missing_token_counts"
    ]
    assert len(findings) == 1
    assert findings[0].locator == {"call_id": "c1"}
    assert findings[0].detail == {"missing": ["completion_tokens"]}


def test_correspondence_reports_only_explicit_capture_contradictions(tmp_path: Path) -> None:
    record = _record(
        0,
        0,
        refs=[{"model_call_id": "missing"}, {"model_call_id": "c1"}],
        usage={"input_tokens": 99, "output_tokens": 99},
    )
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(orjson.dumps(record, option=orjson.OPT_APPEND_NEWLINE))
    capture_dir = tmp_path / "captures"
    capture_dir.mkdir()
    capture_path = capture_dir / "0-0.capture.jsonl"
    raw_exchange = {
        "model_call_id": "c2",
        "status_code": 200,
        "request": {"model": "model"},
        "response": {"id": "r2", "usage": {"input_tokens": 1, "output_tokens": 1}},
    }
    capture_path.write_bytes(
        b"\n"
        + b"[]\n"
        + b"{not-json}\n"
        + orjson.dumps(
            _call(model_call_id="failed", response_id="failed-response", status_code=500),
            option=orjson.OPT_APPEND_NEWLINE,
        )
        + orjson.dumps(_call(), option=orjson.OPT_APPEND_NEWLINE)
        + orjson.dumps(_call(), option=orjson.OPT_APPEND_NEWLINE)
        + orjson.dumps(raw_exchange, option=orjson.OPT_APPEND_NEWLINE)
    )

    result = run_health_checks(rollout_path, capture_dirs=[capture_dir], capture_enabled=True, workers=1)

    kinds = {
        finding.detail.get("kind")
        for finding in result.rollouts[0].findings
        if finding.check == "trajectory_capture_mismatch"
    }
    assert kinds == {"unreadable_capture_records", "missing_captured_call", "duplicated_captured_call"}
    assert not any(finding.check == "model_call_failed" for finding in result.rollouts[0].findings)
    assert {
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "model_call_failed",
        "rollout_token_count_mismatch",
        "model_call_runaway_generation",
    } <= set(result.rollouts[0].unobserved)
    assert result.summary["run"]["stats"]["duplicated_calls"] == {"replayed": 1, "rollouts": 1}
    assert health_checks._call_identity({"response_id": "loose"}) == "response::loose"
    assert health_checks._call_identity({}) is None


def test_correspondence_uses_bound_calls_and_gym_ids_for_replay(tmp_path: Path) -> None:
    record = _record(0, 0, usage={"input_tokens": 3, "output_tokens": 2})
    rollout_path, capture_dir = _write_fixture(
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

    result = run_health_checks(rollout_path, capture_dirs=[capture_dir], capture_enabled=True, workers=1)

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
    rollout_path, capture_dir = _write_fixture(tmp_path, [(record, [_call()])])

    result = run_health_checks(rollout_path, capture_dirs=[capture_dir], capture_enabled=True, workers=1)

    [digest] = result.rollouts
    assert any(finding.check == "trajectory_capture_mismatch" for finding in digest.findings)
    assert "rollout_token_count_mismatch" in digest.unobserved
    assert "model_call_missing_token_counts" not in digest.unobserved
    assert not any(finding.check == "model_call_missing_token_counts" for finding in digest.findings)


def test_call_failures_and_token_mismatches_have_separate_check_ids(tmp_path: Path) -> None:
    rollout_path, capture_dir = _write_fixture(
        tmp_path,
        [
            (
                _record(0, 0, usage={"input_tokens": 99, "output_tokens": 99}),
                [_call(status_code=500, error_category="upstream")],
            )
        ],
    )

    result = run_health_checks(rollout_path, capture_dirs=[capture_dir], capture_enabled=True, workers=1)

    checks = [finding.check for finding in result.rollouts[0].findings]
    assert checks.count("model_call_failed") == 1
    assert checks.count("rollout_token_count_mismatch") == 1
    assert "trajectory_capture_mismatch" not in checks
    failed = next(finding for finding in result.rollouts[0].findings if finding.check == "model_call_failed")
    assert failed.detail == {"status": 500, "error_category": "upstream", "terminal": True}


def test_duplicate_rollout_identity_counts_once_at_task_scope(tmp_path: Path) -> None:
    duplicate = _record(7, 0, answer=None)
    rollout_path = tmp_path / "rollouts.jsonl"
    rollout_path.write_bytes(
        orjson.dumps(duplicate, option=orjson.OPT_APPEND_NEWLINE)
        + orjson.dumps(duplicate, option=orjson.OPT_APPEND_NEWLINE)
    )

    result = run_health_checks(rollout_path, capture_enabled=False, workers=1)

    assert result.summary["run"]["verdicts"] == {"healthy": 0, "unhealthy": 2, "unobserved": 0}
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


def test_zero_token_call_is_flagged_and_nonempty_length_response_is_exempt(tmp_path: Path) -> None:
    rollout_path, capture_dir = _write_fixture(
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

    result = run_health_checks(rollout_path, capture_dirs=[capture_dir], capture_enabled=True, workers=1)

    checks = {finding.check for finding in result.rollouts[0].findings}
    assert "model_call_zero_completion_tokens" in checks
    assert "model_call_runaway_generation" not in checks
    assert health_checks._response_has_content("malformed") is False
    assert health_checks._response_has_content({"content": "visible"}) is True


def test_malformed_records_and_check_failures_become_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_bytes(b"[]\n")
    parsed = run_health_checks(malformed, capture_enabled=False, workers=1)
    [digest] = parsed.rollouts
    assert digest.task_index == 0 and digest.rollout_index == 0
    assert digest.verdict == "unhealthy"
    assert len(digest.findings) == 1
    assert digest.findings[0].check == "record_unreadable"
    assert digest.findings[0].detail["reason"] == "rollout record is unreadable"
    assert set(digest.unobserved) == {
        "rollout_missing_agent_turns",
        "agent_turn_hollow",
        "model_call_zero_completion_tokens",
        "model_call_missing_token_counts",
        "trajectory_capture_mismatch",
        "model_call_failed",
        "rollout_token_count_mismatch",
        "model_call_runaway_generation",
    }

    healthy = tmp_path / "healthy.jsonl"
    healthy.write_bytes(orjson.dumps(_record(0, 0), option=orjson.OPT_APPEND_NEWLINE))

    def broken_check(*args, **kwargs):
        raise TypeError("bad shape")

    monkeypatch.setitem(health_checks._ROLLOUT_CHECKS, "rollout_missing_agent_turns", broken_check)
    checked = run_health_checks(healthy, capture_enabled=False, workers=1)
    finding = next(item for item in checked.rollouts[0].findings if item.check == "record_unreadable")
    assert finding.detail == {
        "reason": "check input is unreadable",
        "failed_check": "rollout_missing_agent_turns",
        "error": "TypeError",
    }
    assert "rollout_missing_agent_turns" in checked.rollouts[0].unobserved


def test_process_pool_success_path_and_run_discovery(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
    class InlinePool:
        def __init__(self, *, max_workers):
            assert max_workers == 2

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def map(self, function, items):
            return map(function, items)

    monkeypatch.setattr(health, "ProcessPoolExecutor", InlinePool)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    rollout_path = run_dir / "custom-name.jsonl"
    rollout_path.write_bytes(
        b"\n"
        + orjson.dumps(_record(0, 0), option=orjson.OPT_APPEND_NEWLINE)
        + orjson.dumps(_record(1, 0), option=orjson.OPT_APPEND_NEWLINE)
    )

    result = health.health_check_run_dir(run_dir, workers=2)

    assert len(result.rollouts) == 2
    assert "2 checked" in capsys.readouterr().out
    file_result = health.health_check_run_dir(rollout_path, workers=1)
    assert len(file_result.rollouts) == 2


def test_input_validation_and_ambiguous_discovery_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one"):
        run_health_checks([], workers=1)
    with pytest.raises(FileNotFoundError, match="Rollout JSONL"):
        run_health_checks(tmp_path / "missing.jsonl", workers=1)
    with pytest.raises(FileNotFoundError, match="Run directory"):
        health.health_check_run_dir(tmp_path / "missing-run", workers=1)

    run_dir = tmp_path / "ambiguous"
    run_dir.mkdir()
    (run_dir / "a.jsonl").write_text("{}\n")
    (run_dir / "b.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="exactly one"):
        health.health_check_run_dir(run_dir, workers=1)

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
