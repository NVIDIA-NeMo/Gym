# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Where rollout rows keep model-call payloads (``model_call_payloads``), and the merge's memory in each mode."""

import copy
import gc
import tracemalloc
from pathlib import Path

import orjson
import pytest

from nemo_gym.config.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
from nemo_gym.models.model_call_capture import CaptureStore
from nemo_gym.rollouts.model_calls import (
    merge_model_call_capture_into_record,
    model_call_payloads_from_config,
    resolve_model_call_payloads,
)
from nemo_gym.rollouts.rollout_collection import enrich_run_result
from nemo_gym.trajectory.trajectory_record import TrajectoryRecord


_POLICY = {"type": "responses_api_models", "name": "policy_model"}


def _exchange(call_id: str, history: list, text: str) -> dict:
    return {
        "model_call_id": call_id,
        "dialect": "responses",
        "model_ref": _POLICY,
        "started_at": float(len(history)),
        "completed_at": float(len(history)) + 0.5,
        "status_code": 200,
        "request": {"model": "policy", "instructions": "Be terse. " * 40, "input": list(history)},
        "response": {
            "id": f"resp-{call_id}",
            "status": "completed",
            "output": [{"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": text}]}],
            "usage": {"input_tokens": 10 * len(history), "output_tokens": 3},
        },
    }


def _capture_session(store: CaptureStore, rollout_id: str, calls: int, *, item_chars: int = 200) -> list[dict]:
    history: list = [{"role": "user", "content": "Solve the task."}]
    written = []
    for index in range(calls):
        history = history + [{"type": "function_call_output", "call_id": f"t{index}", "output": "o" * item_chars}]
        exchange = _exchange(f"c{index}", history, f"step {index}")
        written.append(orjson.loads(orjson.dumps(exchange)))
        store.record(rollout_id, exchange)
    return written


def _record(calls: int) -> dict:
    """A finished rollout whose agent claims every captured call, so turns are built from the capture."""
    return {
        TASK_INDEX_KEY_NAME: 0,
        ROLLOUT_INDEX_KEY_NAME: 0,
        "reward": 1.0,
        "response": {"output": []},
        "ng_agent_observations": {
            "source": "test",
            "records": [
                {
                    "kind": "agent_invocation",
                    "invocation_id": "root",
                    "model_calls": [{"model_call_id": f"c{index}"} for index in range(calls)],
                }
            ],
        },
    }


def _row() -> dict:
    return {"responses_create_params": {"input": []}, TASK_INDEX_KEY_NAME: 0, ROLLOUT_INDEX_KEY_NAME: 0}


def test_the_payload_mode_is_read_from_the_global_config(tmp_path: Path) -> None:
    assert model_call_payloads_from_config({}) == "inline"
    assert model_call_payloads_from_config({"model_call_payloads": "reference"}) == "reference"
    with pytest.raises(ValueError):
        model_call_payloads_from_config({"model_call_payloads": "sometimes"})


@pytest.mark.parametrize("capture_format", ["full", "delta"])
@pytest.mark.parametrize("payloads", ["inline", "reference", "none"])
def test_each_payload_mode_attaches_what_it_promises(tmp_path: Path, capture_format: str, payloads: str) -> None:
    store = CaptureStore(tmp_path, capture_format=capture_format)
    written = _capture_session(store, "0-0", 4)
    record = _record(4)

    merge_model_call_capture_into_record(record, [tmp_path], payloads=payloads)

    calls = record["ng_model_call_capture"]["calls"]
    assert [call["model_call_id"] for call in calls] == ["c0", "c1", "c2", "c3"]
    assert [call["tokens_in"] for call in calls] == [20, 30, 40, 50]
    for index, (call, exchange) in enumerate(zip(calls, written, strict=True)):
        if payloads == "inline":
            assert call["request"] == exchange["request"] and call["response"] == exchange["response"]
            assert "payload_ref" not in call
        elif payloads == "reference":
            assert "request" not in call and "request_raw" not in call
            assert call["response"] == exchange["response"]
            assert call["payload_ref"] == {
                "capture_file": "0-0.capture.jsonl",
                "call_index": index,
                "model_call_id": f"c{index}",
            }
        else:
            assert not {"request", "response", "request_raw", "response_raw", "payload_ref"} & set(call)


@pytest.mark.parametrize("schema_version", ["1.0", "2.0"])
async def test_referenced_rows_resolve_to_the_inline_rows(tmp_path: Path, schema_version: str) -> None:
    """A row that references its requests carries everything else, and the helper restores the rest exactly."""
    store = CaptureStore(tmp_path, capture_format="delta")
    _capture_session(store, "0-0", 5, item_chars=2000)
    inline, referenced = _record(5), _record(5)
    options = dict(capture_dirs=[tmp_path], observability_enabled=True, trajectory_schema_version=schema_version)

    await enrich_run_result(_row(), inline, **options)
    await enrich_run_result(_row(), referenced, model_call_payloads="reference", **options)

    inline_trajectory, referenced_trajectory = inline["ng_trajectory"], referenced["ng_trajectory"]
    assert len(referenced_trajectory["turns"]) == 5
    assert all(call.get("request") is None and call["payload_ref"] for call in referenced_trajectory["model_calls"])
    assert all(turn.get("question") is None and turn["payload_ref"] for turn in referenced_trajectory["turns"])
    # Answers stay on the row: they are each call's own output, and the health checks read them.
    assert [turn["answer"] for turn in referenced_trajectory["turns"]] == [
        turn["answer"] for turn in inline_trajectory["turns"]
    ]
    TrajectoryRecord.model_validate(referenced_trajectory)
    assert len(orjson.dumps(referenced)) * 3 < len(orjson.dumps(inline))

    resolved = resolve_model_call_payloads(referenced_trajectory, [tmp_path])

    def without_refs(trajectory: dict) -> dict:
        trajectory = copy.deepcopy(trajectory)
        for entry in [*trajectory["model_calls"], *trajectory["turns"]]:
            entry.pop("payload_ref", None)
        return trajectory

    assert without_refs(resolved) == inline_trajectory
    # The resolved requests share their common history items.
    first, last = resolved["model_calls"][1]["request"]["input"], resolved["model_calls"][4]["request"]["input"]
    assert first[1] is last[1]


@pytest.mark.parametrize("schema_version", ["1.0", "2.0"])
@pytest.mark.parametrize("model_call_payloads", ["inline", "reference"])
async def test_a_row_without_payloads_names_no_capture_line(
    tmp_path: Path, schema_version: str, model_call_payloads: str
) -> None:
    """``include_payloads=False`` keeps call metadata only, whatever ``model_call_payloads`` says.

    A v2 row reads responses to link tool calls even then, without reading request histories, and keeps neither.
    """
    _capture_session(CaptureStore(tmp_path, capture_format="delta"), "0-0", 3)
    record = _record(3)

    await enrich_run_result(
        _row(),
        record,
        capture_dirs=[tmp_path],
        observability_enabled=True,
        include_payloads=False,
        model_call_payloads=model_call_payloads,
        trajectory_schema_version=schema_version,
    )

    calls = record["ng_model_call_capture"]["calls"]
    assert [call["model_call_id"] for call in calls] == ["c0", "c1", "c2"]
    assert not any({"request", "response", "request_raw", "response_raw", "payload_ref"} & set(call) for call in calls)
    trajectory = record["ng_trajectory"]
    assert all(call.get("request") is None and call.get("response") is None for call in trajectory["model_calls"])
    assert not any("payload_ref" in entry for entry in [*trajectory["model_calls"], *trajectory["turns"]])
    # Turns are built from the responses read: v1 reads none without payloads, v2 reads them for its tool links.
    assert len(trajectory["turns"]) == (3 if schema_version == "2.0" else 0)


async def test_a_reference_to_an_unreadable_line_stays_a_reference(tmp_path: Path) -> None:
    store = CaptureStore(tmp_path, capture_format="delta")
    _capture_session(store, "0-0", 3)
    record = _record(3)
    await enrich_run_result(
        _row(), record, capture_dirs=[tmp_path], observability_enabled=True, model_call_payloads="reference"
    )
    store.delete("0-0")

    resolved = resolve_model_call_payloads(record["ng_trajectory"], [tmp_path])

    assert resolved == record["ng_trajectory"]


async def test_delete_capture_needs_inline_payloads(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="delete_capture"):
        await enrich_run_result(
            _row(),
            _record(0),
            capture_dirs=[tmp_path],
            observability_enabled=True,
            model_call_payloads="reference",
            delete_capture=True,
        )


def _peak(function) -> int:
    gc.collect()
    tracemalloc.start()
    try:
        function()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


def test_the_merge_holds_one_call_at_a_time_unless_it_keeps_payloads(tmp_path: Path) -> None:
    """The finished-rollout merge reads request histories only when the row keeps them, and shares them when it does.

    The baseline is what the merge used to do: parse every line of the capture into its own objects.
    """
    full, delta = CaptureStore(tmp_path / "full"), CaptureStore(tmp_path / "delta", capture_format="delta")
    _capture_session(full, "0-0", 100, item_chars=4000)
    _capture_session(delta, "0-0", 100, item_chars=4000)
    lines = full.path_for("0-0").read_bytes().splitlines()
    every_line_parsed = _peak(lambda: [orjson.loads(line) for line in lines])

    def merge_peak(directory: Path, payloads: str) -> int:
        return _peak(lambda: merge_model_call_capture_into_record(_record(100), [directory], payloads=payloads))

    for directory in (tmp_path / "full", tmp_path / "delta"):
        for payloads in ("reference", "none"):
            assert merge_peak(directory, payloads) * 4 < every_line_parsed
    # Keeping the payloads holds the history once, since consecutive calls share their items.
    assert merge_peak(tmp_path / "delta", "inline") * 4 < every_line_parsed
