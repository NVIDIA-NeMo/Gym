# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import sys
from asyncio import Future
from pathlib import Path

import orjson
import pytest

import nemo_gym.cli.eval as cli_eval
import nemo_gym.cli.main as cli_main
import nemo_gym.rollout_collection as rollout_collection
import nemo_gym.rollout_health as health
from nemo_gym.rollout_collection import RolloutCollectionConfig, RolloutCollectionHelper
from nemo_gym.rollout_health import CHECK_REGISTRY, run_health_checks


CAPTURE_CHECKS = {
    "zero_token_turns",
    "missed_metrics",
    "transcript_capture_correspondence",
    "runaway_generations",
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


def test_all_eight_v1_checks_fire_on_synthetic_artifacts(tmp_path: Path) -> None:
    rows = [
        (_record(0, 0, include_turn=False, include_response_output=False), [_call()]),
        (_record(0, 1, include_turn=False, include_response_output=False), [_call()]),
        (_record(1, 0, answer=None), [_call()]),
        (
            _record(2, 0, usage={"input_tokens": 3, "output_tokens": 0}),
            [_call(tokens_out=0, response={"output_text": ""})],
        ),
        (_record(3, 0, usage=None), [_call(tokens_out=None)]),
        (_record(4, 0, usage=None), [_call(status_code=500, error_category="upstream")]),
        (
            _record(5, 0, usage={"input_tokens": 3, "output_tokens": 2}),
            [_call(finish_reason="length", response={})],
        ),
        (_record(6, 0, usage=None), [_call(status_code=500)]),
        (_record(6, 1, usage=None), [_call(status_code=408)]),
    ]
    rollout_path, capture_dir = _write_fixture(tmp_path, rows)

    result = run_health_checks(rollout_path, capture_dirs=[capture_dir], capture_enabled=True, workers=2)

    assert set(result.summary["run"]["issues"]) == {spec.id for spec in CHECK_REGISTRY}
    assert all(
        result.summary["run"]["issues"][spec.id] > 0 for spec in CHECK_REGISTRY if spec.id != "unreadable_record"
    )
    assert result.summary["run"]["issues"]["unreadable_record"] == 0
    assert result.summary["tasks"]["0"]["flags"] == ["consistently_unhealthy_task"]
    assert "no_healthy_model_calls_task" in result.summary["tasks"]["6"]["flags"]
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


def test_health_check_cli_accepts_run_dir_and_workers(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    received = {}

    def fake_health_check(run_dir, *, workers=None):
        received.update(run_dir=run_dir, workers=workers)

    monkeypatch.setattr(cli_eval, "health_check_rollouts", fake_health_check)
    monkeypatch.setattr(sys, "argv", ["gym", "eval", "health-check", str(tmp_path), "--workers", "3"])

    cli_main.main()

    assert received == {"run_dir": str(tmp_path), "workers": 3}


def test_persisted_trajectory_and_invocation_conversation_are_valid_capture_inputs(tmp_path: Path) -> None:
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

    result = run_health_checks(rollout_path, workers=1)

    [digest] = result.rollouts
    assert digest.rollout_id == "explicit-rollout"
    assert digest.capture_observed
    assert digest.model_calls == 1
    assert digest.capture_prompt_tokens == 4
    assert not any(finding.check == "missing_agent_steps" for finding in digest.findings)
    assert health._normalized_embedded_calls({"ng_model_call_capture": {"calls": [None, {"call_index": 1}]}}) == [
        {"call_index": 1}
    ]
    assert health._nonempty(123) is False
    assert health._item_has_tool_call("bad") is False
    assert health._item_is_agent_content("bad") is False


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

    result = run_health_checks(rollout_path, capture_enabled=False, workers=1)

    steps = health._agent_steps(record)
    assert len(steps) == 2
    assert steps[0].has_message and steps[0].has_tool_calls
    assert steps[1].has_message and not steps[1].has_tool_calls
    assert not any(finding.check == "hollow_steps" for finding in result.rollouts[0].findings)


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
    assert digest.unobserved == ["missed_metrics"]
    assert digest.verdict == "unobserved"
    assert not any(finding.check in {"hollow_steps", "missed_metrics"} for finding in digest.findings)


def test_missing_one_binding_is_a_missed_metrics_finding(tmp_path: Path) -> None:
    record = _record(0, 0)
    record["ng_trajectory"]["turns"].append(
        {
            "turn_no": 2,
            "answer": "second answer",
            "model_calls": [],
        }
    )
    rollout_path, capture_dir = _write_fixture(tmp_path, [(record, [_call()])])

    result = run_health_checks(rollout_path, capture_dirs=[capture_dir], capture_enabled=True, workers=1)

    missed = [finding for finding in result.rollouts[0].findings if finding.check == "missed_metrics"]
    assert len(missed) == 1
    assert missed[0].locator == {"turn": 2}
    assert missed[0].detail["reason"] == "transcript step has no call binding"


def test_correspondence_handles_raw_damaged_replayed_and_retried_capture_lines(tmp_path: Path) -> None:
    record = _record(0, 0, usage={"input_tokens": 99, "output_tokens": 99})
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
        if finding.check == "transcript_capture_correspondence"
    }
    assert {
        "unreadable_capture_records",
        "call_count_delta",
        "duplicated_call",
        "failed_call",
        "retry_dropped_call",
        "token_sum_mismatch",
    } <= kinds
    assert result.summary["run"]["stats"]["duplicated_calls"] == {"replayed": 1, "rollouts": 1}
    assert health._call_identity({"response_id": "loose"}) == "response::loose"
    assert health._call_identity({}) is None


def test_explicit_deterministic_dispatch_and_nonempty_length_response_are_exempt(tmp_path: Path) -> None:
    rollout_path, capture_dir = _write_fixture(
        tmp_path,
        [
            (
                _record(0, 0, usage={"input_tokens": 3, "output_tokens": 0}),
                [
                    _call(
                        tokens_out=0,
                        finish_reason="length",
                        request={"metadata": {"deterministic_dispatch": True}},
                        response={"choices": [{"message": {"content": "kept"}}]},
                    )
                ],
            )
        ],
    )

    result = run_health_checks(rollout_path, capture_dirs=[capture_dir], capture_enabled=True, workers=1)

    checks = {finding.check for finding in result.rollouts[0].findings}
    assert "zero_token_turns" not in checks
    assert "runaway_generations" not in checks
    assert health._is_deterministic_dispatch({"request": "malformed"}) is False
    assert health._response_has_content("malformed") is False
    assert health._response_has_content({"content": "visible"}) is True


def test_malformed_records_and_check_failures_become_findings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    malformed = tmp_path / "malformed.jsonl"
    malformed.write_bytes(b"[]\n")
    parsed = run_health_checks(malformed, capture_enabled=False, workers=1)
    [digest] = parsed.rollouts
    assert digest.task_index == 0 and digest.rollout_index == 0
    assert digest.verdict == "unhealthy"
    assert len(digest.findings) == 1
    assert digest.findings[0].check == "unreadable_record"
    assert digest.findings[0].detail["reason"] == "rollout record is unreadable"
    assert set(digest.unobserved) == {
        "missing_agent_steps",
        "hollow_steps",
        "zero_token_turns",
        "missed_metrics",
        "transcript_capture_correspondence",
        "runaway_generations",
    }

    healthy = tmp_path / "healthy.jsonl"
    healthy.write_bytes(orjson.dumps(_record(0, 0), option=orjson.OPT_APPEND_NEWLINE))

    def broken_check(*args, **kwargs):
        raise TypeError("bad shape")

    monkeypatch.setitem(health._ROLLOUT_CHECKS, "missing_agent_steps", broken_check)
    checked = run_health_checks(healthy, capture_enabled=False, workers=1)
    finding = next(item for item in checked.rollouts[0].findings if item.check == "unreadable_record")
    assert finding.detail == {
        "reason": "check input is unreadable",
        "failed_check": "missing_agent_steps",
        "error": "TypeError",
    }
    assert "missing_agent_steps" in checked.rollouts[0].unobserved


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
