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
"""Gate unit tests: custody over the base LineageIndex, capacity, causes.

The HTTP-level behavior lives in the vllm_model app test; this file covers
the gate object directly -- serving-rule causes, coords ingestion into the
index, seal/TTL state drops, and the capacity/eviction accounting that the
migration review flagged as load-bearing (finding M).
"""

import pytest

from nemo_gym.token_id_capture.gate import RolloutCaptureGate, RolloutGateConfig
from nemo_gym.token_id_capture.staging.lineage import DuplicateRolloutError, UnknownRolloutError
from nemo_gym.token_id_capture.staging.records import CommitCoords, staging_key


def _coords(rollout_id: str, call_id: str, parent: str | None, prev_len: int, delta: list[int]) -> CommitCoords:
    return CommitCoords(
        rollout_id=rollout_id,
        call_id=call_id,
        parent_call_id=parent,
        delta_len=len(delta),
        cum_len=prev_len + len(delta),
        digest="d-" + call_id,
        staging_key=staging_key(rollout_id, call_id),
        weight_version=1,
        token_ids_delta=delta,
    )


def _serve_turn(gate: RolloutCaptureGate, rollout_id: str, call_id: str, messages: list[dict], content: str):
    """One full call through the gate object: admit, then commit with a
    served assistant turn, as the app does around the engine call."""
    decision = gate.prepare_call(rollout_id, call_id, messages)
    prev_len = decision.prev_len
    coords = _coords(rollout_id, call_id, decision.parent_call_id, prev_len, [prev_len + 1, prev_len + 2])
    gate.ingest_coords(coords, request_messages=messages, served_turn={"role": "assistant", "content": content})
    return decision


def test_serving_rule_causes_and_chaining() -> None:
    gate = RolloutCaptureGate()
    gate.register_rollout("r0")

    # First call: nothing recorded, not a fallback.
    first = _serve_turn(gate, "r0", "c1", [{"role": "user", "content": "hi"}], "turn 1")
    assert first.mode == "text" and first.fallback_reason is None

    # Echoed history resolves to c1 with its exact cumulative ids.
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "turn 1"},
        {"role": "user", "content": "next"},
    ]
    second = gate.prepare_call("r0", "c2", history)
    assert second.mode == "token_in"
    assert second.parent_call_id == "c1"
    assert second.prefix_ids == [1, 2]
    assert second.prev_len == 2

    # A request with no assistant turns after calls were recorded: no_history.
    third = gate.prepare_call("r0", "c3", [{"role": "user", "content": "fresh side conversation"}])
    assert third.mode == "text" and third.fallback_reason == "no_history"

    # A rewritten assistant turn: no_match.
    rewritten = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "REWRITTEN"},
        {"role": "user", "content": "next"},
    ]
    fourth = gate.prepare_call("r0", "c4", rewritten)
    assert fourth.mode == "text" and fourth.fallback_reason == "no_match"

    metrics = gate.snapshot_metrics()
    assert metrics["token_in"] == 1
    assert metrics["fallback_no_history"] == 1
    assert metrics["fallback_no_match"] == 1


def test_ambiguous_turns_fall_back() -> None:
    gate = RolloutCaptureGate()
    gate.register_rollout("r0")
    _serve_turn(gate, "r0", "c1", [{"role": "user", "content": "hi"}], "SAME")
    _serve_turn(gate, "r0", "c2", [{"role": "user", "content": "hi"}], "SAME")
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "SAME"},
        {"role": "user", "content": "next"},
    ]
    decision = gate.prepare_call("r0", "c3", history)
    assert decision.mode == "text" and decision.fallback_reason == "ambiguous"
    assert gate.snapshot_metrics()["fallback_ambiguous"] == 1


def test_capture_failed_is_not_indexed() -> None:
    gate = RolloutCaptureGate()
    gate.register_rollout("r0")
    decision = gate.prepare_call("r0", "c1", [{"role": "user", "content": "hi"}])
    failed = _coords("r0", "c1", decision.parent_call_id, 0, [])
    failed = failed.model_copy(update={"disposition": "capture_failed", "delta_len": 0, "cum_len": 0})
    assert (
        gate.ingest_coords(
            failed,
            request_messages=[{"role": "user", "content": "hi"}],
            served_turn={"role": "assistant", "content": "x"},
        )
        is False
    )
    # Nothing indexed: a child echoing the turn cannot chain onto it.
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "x"},
        {"role": "user", "content": "next"},
    ]
    child = gate.prepare_call("r0", "c2", history)
    assert child.mode == "text"
    assert gate.snapshot_metrics()["capture_failed"] == 1


def test_seal_and_expiry_drop_index_state() -> None:
    gate = RolloutCaptureGate()
    gate.register_rollout("r0")
    _serve_turn(gate, "r0", "c1", [{"role": "user", "content": "hi"}], "turn 1")
    assert gate.snapshot_metrics()["lineage_rollouts"] == 1
    assert gate.snapshot_metrics()["lineage_tokens"] > 0
    # A duplicate registration of a LIVE rollout is the rerun-protection case.
    with pytest.raises(DuplicateRolloutError):
        gate.register_rollout("r0")

    receipt = gate.seal_rollout("r0", reward=1.0)
    assert receipt.terminal_call_id == "c1"
    assert gate.snapshot_metrics()["lineage_rollouts"] == 0
    with pytest.raises(UnknownRolloutError):
        gate.prepare_call("r0", "c9", [])


def test_eviction_is_counted(monkeypatch) -> None:
    gate = RolloutCaptureGate(lineage_max_rollouts=1, lineage_max_tokens=10)
    gate.register_rollout("r0")
    gate.register_rollout("r1")
    _serve_turn(gate, "r0", "c1", [{"role": "user", "content": "a"}], "turn a")
    _serve_turn(gate, "r1", "c2", [{"role": "user", "content": "b"}], "turn b")
    metrics = gate.snapshot_metrics()
    assert metrics["lineage_evictions"] >= 1, "capacity pressure must be loud, not silent"


def test_install_lineage_index_replaces_module_global() -> None:
    from nemo_gym.token_id_capture import sink as sink_module
    from nemo_gym.token_id_capture.sink import lineage_index

    previous = sink_module._LINEAGE
    try:
        gate = RolloutCaptureGate.from_config(
            RolloutGateConfig(enabled=True, lineage_max_rollouts=7, control_auth_token="t")
        )
        gate.install_lineage_index()
        assert lineage_index() is gate._index  # the #2180-c1 workaround contract
    finally:
        sink_module._LINEAGE = previous
