# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""S3: the gate — marker resolution, serving rule, coords ingestion, receipts.

The centerpiece replays the S1 golden call sequences *through the gate*
(agent-echo simulation: histories carry markers, a fake worker turns
decisions into staged coords) and requires the sealed receipt to be
byte-identical to the direct lineage drive of the same call sequence — the
gate adds custody, never semantics. Around it: the serving-rule fallback
matrix, duplicate/wrong-rollout coords rejection, TTL expiry, the in-memory
token buffer's fork-aware prefix serving, and the control routes.
"""

from typing import Any, Optional

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nemo_gym.token_id_capture.control_routes import install_rollout_control_routes
from nemo_gym.token_id_capture.gate import (
    NG_CALL_ID_FIELD,
    RolloutCaptureGate,
    find_marker,
    message_fingerprint,
)
from nemo_gym.token_id_capture.memory_store import MemoryRolloutTokenBuffer, TokenBufferError
from nemo_gym.token_id_capture.staging.conformance import fixture_names, load_fixture
from nemo_gym.token_id_capture.staging.conformance.kit import build_fixture_artifacts, f32
from nemo_gym.token_id_capture.staging.digest import build_staging_delta, compute_staging_digest
from nemo_gym.token_id_capture.staging.lineage import (
    DuplicateRolloutError,
    LineageStateError,
    UnknownCallError,
    UnknownRolloutError,
)
from nemo_gym.token_id_capture.staging.records import CommitCoords, staging_key


def _worker_coords(
    decision: Any, *, prompt_ids: list[int], gen_ids: list[int], gen_lps: list[float], wv: int
) -> CommitCoords:
    """What the capture-enabled worker stages and rides back on the response."""
    ids_delta, mask_delta, lp_delta = build_staging_delta(
        prompt_token_ids=prompt_ids,
        generated_token_ids=gen_ids,
        generated_logprobs=gen_lps,
        prev_len=decision.prev_len,
    )
    digest = compute_staging_digest(
        rollout_id=decision.rollout_id,
        call_id=decision.call_id,
        prev_len=decision.prev_len,
        token_ids_delta=ids_delta,
        token_mask_delta=mask_delta,
        logprobs_delta=lp_delta,
    )
    return CommitCoords(
        rollout_id=decision.rollout_id,
        call_id=decision.call_id,
        parent_call_id=decision.parent_call_id,
        delta_len=len(ids_delta),
        cum_len=decision.prev_len + len(ids_delta),
        digest=digest,
        staging_key=staging_key(decision.rollout_id, decision.call_id),
        weight_version=wv,
        token_ids_delta=ids_delta,
    )


def _failed_coords(decision: Any, *, wv: int) -> CommitCoords:
    return CommitCoords(
        rollout_id=decision.rollout_id,
        call_id=decision.call_id,
        parent_call_id=decision.parent_call_id,
        delta_len=0,
        cum_len=decision.prev_len,
        digest="",
        staging_key=staging_key(decision.rollout_id, decision.call_id),
        weight_version=wv,
        disposition="capture_failed",
    )


def _replay_fixture_through_gate(fixture: dict) -> tuple[Any, dict[str, str]]:
    """Drive one golden call sequence through the gate as agent + worker.

    Marker-linked histories reproduce each call's fixture parent: a child of
    ``p`` sends exactly the history recorded when ``p`` was served (echo), a
    ``text``-mode non-root call sends an *edited* history (fingerprint miss).
    Returns the sealed receipt and the fixture->gate call-id mapping.
    """
    rollout_id = fixture["rollout_id"]
    gate = RolloutCaptureGate()
    gate.register_rollout(rollout_id)

    id_map: dict[str, str] = {}
    # Per fixture call: the exact history a child of that call would echo.
    echo_history: dict[str, list[dict]] = {}
    terminal_gate_id: Optional[str] = None

    for index, call in enumerate(fixture["calls"]):
        parent = call.get("parent_call_id")
        mode = call.get("mode", "token_in" if parent else "text")
        if parent is not None:
            messages = [dict(m) for m in echo_history[parent]]
            messages.append({"role": "user", "content": f"continue {index}"})
        elif index == 0:
            messages = [{"role": "user", "content": "start"}]
        else:
            # A non-first root is the fingerprint-miss/edited-history case:
            # echo the first call's marker but rewrite content above it.
            first = fixture["calls"][0]["call_id"]
            messages = [dict(m) for m in echo_history[first]]
            messages[0] = {**messages[0], "content": "REWRITTEN"}
            messages.append({"role": "user", "content": f"continue {index}"})

        decision = gate.prepare_call(rollout_id, messages)
        id_map[call["call_id"]] = decision.call_id

        if mode == "token_in":
            assert decision.mode == "token_in", f"{call['call_id']}: expected token-in, got {decision.fallback_reason}"
            assert decision.parent_call_id == id_map[parent]
            if "prompt_token_ids" in call:  # absent on capture-failed calls
                prompt_ids = [int(t) for t in call["prompt_token_ids"]]
                assert decision.prefix_ids == prompt_ids[: decision.prev_len], "served prefix diverges from exact ids"
        else:
            assert decision.mode == "text" and decision.prefix_ids is None

        if call.get("disposition") == "capture_failed":
            coords = _failed_coords(decision, wv=int(call["weight_version"]))
            marker = gate.ingest_coords(coords, request_messages=messages, served_message={"role": "assistant"})
            assert marker is None, "capture-failed calls must not release a marker"
            continue

        coords = _worker_coords(
            decision,
            prompt_ids=[int(t) for t in call["prompt_token_ids"]],
            gen_ids=[int(t) for t in call["generation_token_ids"]],
            gen_lps=[f32(p) for p in call["generation_log_probs"]],
            wv=int(call["weight_version"]),
        )
        served = {"role": "assistant", "content": f"served {call['call_id']}"}
        marker = gate.ingest_coords(coords, request_messages=messages, served_message=served)
        assert marker == decision.call_id
        echo_history[call["call_id"]] = [*messages, {**served, NG_CALL_ID_FIELD: marker}]
        if call["call_id"] == fixture.get("terminal_call_id"):
            terminal_gate_id = marker

    receipt = gate.seal_rollout(rollout_id, reward=fixture.get("reward"), terminal_call_id=terminal_gate_id)
    return receipt, id_map


@pytest.mark.parametrize("name", fixture_names())
def test_gate_replay_matches_direct_lineage_drive(name: str) -> None:
    """The S3 conformance statement: the same call sequence produces a
    byte-identical receipt whether driven through the gate (markers,
    fingerprints, prefix serving) or directly through RolloutLineage."""
    fixture = load_fixture(name)
    receipt, id_map = _replay_fixture_through_gate(fixture)

    renamed = dict(fixture)
    renamed["calls"] = [
        {**call, "call_id": id_map[call["call_id"]], "parent_call_id": id_map.get(call.get("parent_call_id"))}
        for call in fixture["calls"]
    ]
    if renamed.get("terminal_call_id"):
        renamed["terminal_call_id"] = id_map[renamed["terminal_call_id"]]
    _, _, direct_receipt, _ = build_fixture_artifacts(renamed)

    assert receipt.model_dump() == direct_receipt.model_dump()


# ---------------------------------------------------------------------------
# serving rule: the fallback matrix
# ---------------------------------------------------------------------------


def _one_committed_call(gate: RolloutCaptureGate, rollout_id: str = "r0") -> tuple[list[dict], str]:
    gate.register_rollout(rollout_id)
    messages = [{"role": "user", "content": "hi"}]
    decision = gate.prepare_call(rollout_id, messages)
    coords = _worker_coords(decision, prompt_ids=[10, 11], gen_ids=[12, 13], gen_lps=[-0.1, -0.2], wv=1)
    served = {"role": "assistant", "content": "hello"}
    marker = gate.ingest_coords(coords, request_messages=messages, served_message=served)
    return [*messages, {**served, NG_CALL_ID_FIELD: marker}], marker


def test_happy_path_child_is_token_in_with_exact_prefix() -> None:
    gate = RolloutCaptureGate()
    history, marker = _one_committed_call(gate)
    decision = gate.prepare_call("r0", [*history, {"role": "user", "content": "next"}])
    assert decision.mode == "token_in"
    assert decision.parent_call_id == marker
    assert decision.prefix_ids == [10, 11, 12, 13]
    assert decision.prev_len == 4
    assert gate.metrics["token_in"] == 1


def test_stripped_marker_falls_back_to_text_root() -> None:
    gate = RolloutCaptureGate()
    history, _ = _one_committed_call(gate)
    stripped = [{k: v for k, v in m.items() if k != NG_CALL_ID_FIELD} for m in history]
    decision = gate.prepare_call("r0", [*stripped, {"role": "user", "content": "next"}])
    assert decision.mode == "text" and decision.prefix_ids is None
    assert decision.fallback_reason == "no_marker"
    assert gate.metrics["fallback_no_marker"] == 1


def test_edited_history_above_marker_falls_back() -> None:
    gate = RolloutCaptureGate()
    history, _ = _one_committed_call(gate)
    edited = [dict(m) for m in history]
    edited[0]["content"] = "hi, edited by the framework"
    decision = gate.prepare_call("r0", [*edited, {"role": "user", "content": "next"}])
    assert decision.mode == "text"
    assert decision.fallback_reason == "fingerprint_miss"
    assert gate.metrics["fallback_fingerprint_miss"] == 1


def test_unknown_marker_falls_back() -> None:
    gate = RolloutCaptureGate()
    history, _ = _one_committed_call(gate)
    forged = [dict(m) for m in history]
    forged[-1][NG_CALL_ID_FIELD] = "deadbeef"
    decision = gate.prepare_call("r0", [*forged, {"role": "user", "content": "next"}])
    assert decision.mode == "text"
    assert decision.fallback_reason == "unknown_marker"


def test_reasoning_stripping_survives_fingerprinting() -> None:
    """History renders drop <think> blocks; the fingerprint must not care."""
    gate = RolloutCaptureGate()
    gate.register_rollout("r0")
    messages = [{"role": "user", "content": "hi"}]
    decision = gate.prepare_call("r0", messages)
    coords = _worker_coords(decision, prompt_ids=[1], gen_ids=[2], gen_lps=[-0.5], wv=1)
    served = {"role": "assistant", "content": "<think>secret plan</think>the answer"}
    marker = gate.ingest_coords(coords, request_messages=messages, served_message=served)
    echoed = [*messages, {"role": "assistant", "content": "the answer", NG_CALL_ID_FIELD: marker}]
    decision = gate.prepare_call("r0", [*echoed, {"role": "user", "content": "next"}])
    assert decision.mode == "token_in"


def test_unregistered_rollout_is_a_loud_contract_violation() -> None:
    gate = RolloutCaptureGate()
    with pytest.raises(UnknownRolloutError):
        gate.prepare_call("never-registered", [{"role": "user", "content": "hi"}])


def test_create_only_registration() -> None:
    gate = RolloutCaptureGate()
    gate.register_rollout("r0")
    with pytest.raises(DuplicateRolloutError):
        gate.register_rollout("r0")


# ---------------------------------------------------------------------------
# coords ingestion: rejection + poisoning
# ---------------------------------------------------------------------------


def test_duplicate_coords_are_rejected() -> None:
    gate = RolloutCaptureGate()
    gate.register_rollout("r0")
    messages = [{"role": "user", "content": "hi"}]
    decision = gate.prepare_call("r0", messages)
    coords = _worker_coords(decision, prompt_ids=[1], gen_ids=[2], gen_lps=[-0.5], wv=1)
    gate.ingest_coords(coords, request_messages=messages, served_message={"role": "assistant", "content": "x"})
    with pytest.raises(LineageStateError, match="cannot commit"):
        gate.ingest_coords(coords, request_messages=messages, served_message={"role": "assistant", "content": "x"})


def test_wrong_rollout_coords_are_rejected() -> None:
    gate = RolloutCaptureGate()
    gate.register_rollout("r0")
    gate.register_rollout("r1")
    decision = gate.prepare_call("r0", [{"role": "user", "content": "hi"}])
    coords = _worker_coords(decision, prompt_ids=[1], gen_ids=[2], gen_lps=[-0.5], wv=1)
    forged = coords.model_copy(update={"rollout_id": "r1"})
    # r1 has no such admitted call: the forgery is rejected loudly.
    with pytest.raises(UnknownCallError):
        gate.ingest_coords(forged, request_messages=[], served_message={"role": "assistant"})


def test_capture_failed_coords_poison_and_release_no_marker() -> None:
    gate = RolloutCaptureGate()
    gate.register_rollout("r0")
    messages = [{"role": "user", "content": "hi"}]
    decision = gate.prepare_call("r0", messages)
    marker = gate.ingest_coords(
        _failed_coords(decision, wv=1), request_messages=messages, served_message={"role": "assistant"}
    )
    assert marker is None
    receipt = gate.seal_rollout("r0", reward=0.0)
    assert receipt.capture_poisoned and receipt.manifest == []


def test_seal_drops_all_gate_state_and_ttl_sweeps() -> None:
    gate = RolloutCaptureGate(registration_ttl_s=10.0)
    history, _ = _one_committed_call(gate)
    gate.seal_rollout("r0", reward=1.0)
    assert not gate.is_registered("r0")
    # A child arriving after seal cannot resolve anything.
    with pytest.raises(UnknownRolloutError):
        gate.prepare_call("r0", history)
    # TTL backstop for a rollout whose controller died.
    gate.register_rollout("r1")
    gate.lineage("r1").updated_at -= 60.0
    assert gate.expire_stale() == ["r1"]
    assert not gate.is_registered("r1")


# ---------------------------------------------------------------------------
# memory buffer: fork-aware prefix serving
# ---------------------------------------------------------------------------


def test_buffer_serves_fork_prefixes_from_the_delta_forest() -> None:
    buffer = MemoryRolloutTokenBuffer()
    buffer.register("r0")
    buffer.add_call("r0", "c1", parent_call_id=None, token_ids_delta=[10, 11, 12, 13, 14])
    buffer.add_call("r0", "c2", parent_call_id="c1", token_ids_delta=[20, 21, 22, 23, 24])
    buffer.add_call("r0", "c3", parent_call_id="c1", token_ids_delta=[30, 31])
    assert buffer.cumulative_ids("r0", "c1") == [10, 11, 12, 13, 14]
    assert buffer.cumulative_ids("r0", "c2") == [10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
    assert buffer.cumulative_ids("r0", "c3") == [10, 11, 12, 13, 14, 30, 31]
    assert buffer.total_tokens("r0") == 12  # deltas, not duplicated prefixes
    buffer.drop("r0")
    assert "r0" not in buffer


def test_buffer_contract_violations_are_loud() -> None:
    buffer = MemoryRolloutTokenBuffer()
    buffer.register("r0")
    with pytest.raises(TokenBufferError, match="not buffered"):
        buffer.add_call("r0", "c2", parent_call_id="ghost", token_ids_delta=[1])
    with pytest.raises(TokenBufferError, match="empty delta"):
        buffer.add_call("r0", "c1", parent_call_id=None, token_ids_delta=[])
    with pytest.raises(TokenBufferError, match="create-only"):
        buffer.register("r0")


# ---------------------------------------------------------------------------
# control routes
# ---------------------------------------------------------------------------


def _control_client() -> tuple[TestClient, RolloutCaptureGate]:
    gate = RolloutCaptureGate()
    app = FastAPI()
    install_rollout_control_routes(app, gate)
    return TestClient(app), gate


def test_control_register_seal_roundtrip() -> None:
    client, gate = _control_client()
    assert client.put("/ng-control/rollouts/r0").status_code == 200
    assert client.put("/ng-control/rollouts/r0").status_code == 409  # create-only

    messages = [{"role": "user", "content": "hi"}]
    decision = gate.prepare_call("r0", messages)
    coords = _worker_coords(decision, prompt_ids=[1, 2], gen_ids=[3], gen_lps=[-0.5], wv=7)
    gate.ingest_coords(coords, request_messages=messages, served_message={"role": "assistant", "content": "x"})

    response = client.post("/ng-control/rollouts/r0/seal", json={"reward": 0.5})
    assert response.status_code == 200
    receipt = response.json()
    assert receipt["rollout_id"] == "r0"
    assert receipt["reward"] == 0.5
    assert [entry["weight_version"] for entry in receipt["manifest"]] == [7]
    assert client.post("/ng-control/rollouts/r0/seal", json={}).status_code == 404  # state dropped


def test_control_fail_is_idempotent() -> None:
    client, _ = _control_client()
    assert client.put("/ng-control/rollouts/r0").status_code == 200
    assert client.post("/ng-control/rollouts/r0/fail", json={"reason": "abort"}).json()["failed"] is True
    assert client.post("/ng-control/rollouts/r0/fail", json={"reason": "abort"}).json()["failed"] is False
    assert client.get("/ng-control/metrics").json()["failed_rollouts"] == 1


# ---------------------------------------------------------------------------
# marker survival through the responses converter (agent echo round trip)
# ---------------------------------------------------------------------------


def test_marker_rides_the_converter_round_trip() -> None:
    from nemo_gym.openai_utils import (
        NeMoGymChatCompletion,
        NeMoGymResponseCreateParamsNonStreaming,
    )
    from nemo_gym.responses_converter import VLLMConverter

    converter = VLLMConverter(return_token_id_information=False, uses_reasoning_parser=False)

    # Serve: chat message carrying the marker -> responses output items.
    chat = NeMoGymChatCompletion.model_validate(
        {
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": "hello", NG_CALL_ID_FIELD: "abc123"},
                }
            ],
        }
    )
    params = NeMoGymResponseCreateParamsNonStreaming(input="hi", model="gpt-4o")
    response = converter.chat_completion_to_response(responses_create_params=params, chat_completion=chat)
    assert response.output[-1].ng_call_id == "abc123"

    # Echo: the agent feeds history (with the marker item) back in.
    echoed = NeMoGymResponseCreateParamsNonStreaming(
        input=[
            {"role": "user", "content": "hi", "type": "message"},
            *[item.model_dump() for item in response.output],
            {"role": "user", "content": "next", "type": "message"},
        ]
    )
    chat_params = converter.responses_to_chat_completion_create_params(echoed)
    messages = [dict(m) for m in chat_params.messages]
    marker, index = find_marker(messages)
    assert marker == "abc123"
    assert messages[index]["role"] == "assistant"


def test_fingerprint_ignores_capture_carriers() -> None:
    base = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "yo"}]
    with_carriers = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "yo",
            NG_CALL_ID_FIELD: "abc",
            "prompt_token_ids": [1],
            "generation_token_ids": [2],
            "generation_log_probs": [-0.1],
        },
    ]
    assert message_fingerprint(base) == message_fingerprint(with_carriers)
