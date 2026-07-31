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
"""Gate-mode VLLMModel end-to-end over HTTP (fake vLLM worker), on the stack.

Drives the model server exactly as the RL controller + agent + worker would:
register over the (bearer-authed) control route, run a two-turn chat
conversation through the ``/ng-rollout/<id>`` prefix where the fake worker
honors ``required_prefix_token_ids`` and returns ``CommitCoords``, then seal
and check the receipt. Identity is the stack's: the capture middleware strips
the prefix and mints ``model_call_id``; lineage is the base content
fingerprint (no marker). Asserts the wire properties the design promises:
exact prefix service, no token arrays / logprobs on the agent-facing
response, coords ingestion as the commit, capture-failed poisoning when a
response carries no coords, and loud rejection of prefixed-but-unknown ids.
"""

from typing import Any, Optional
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import nemo_gym.server_utils
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.staging.digest import build_staging_delta, compute_staging_digest
from nemo_gym.token_id_capture.staging.records import staging_key
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


AUTH = {"Authorization": "Bearer test-control-token"}


class _FakeVLLMWorker:
    """A capture-enabled worker: splices the served prefix, 'generates' fixed
    ids, stages the delta (digest recorded), and rides coords on the response."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.staged: list[str] = []
        self.attach_coords = True
        self.fixed_content: Optional[str] = None
        self._turn = 0

    async def create_chat_completion(self, **body: Any) -> dict:
        self.requests.append(body)
        context = body.get("ng_capture")
        prefix = body.get("required_prefix_token_ids") or []
        self._turn += 1
        if context is not None and context["mode"] == "token_in":
            prompt_ids = list(prefix) + [100 + self._turn, 101 + self._turn]  # spliced suffix render
        else:
            prompt_ids = [10, 11, 12]  # full template render
        generated = [200 + self._turn, 201 + self._turn]
        logprobs = [-0.25, -0.5]

        content = self.fixed_content if self.fixed_content is not None else f"turn {self._turn}"
        choice: dict = {
            "index": 0,
            "finish_reason": "stop",
            "message": {"role": "assistant", "content": content},
        }
        if body.get("logprobs"):
            # What return_tokens_as_token_ids puts on the wire; the gate must
            # strip it before the agent sees the response.
            choice["logprobs"] = {
                "content": [{"token": f"token_id:{t}", "logprob": lp} for t, lp in zip(generated, logprobs)]
            }
        response: dict = {
            "id": "chtcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "dummy"),
            "choices": [choice],
        }
        if context is not None and self.attach_coords:
            ids_delta, mask_delta, lp_delta = build_staging_delta(
                prompt_token_ids=prompt_ids,
                generated_token_ids=generated,
                generated_logprobs=logprobs,
                prev_len=context["prev_len"],
            )
            digest = compute_staging_digest(
                rollout_id=context["rollout_id"],
                call_id=context["call_id"],
                prev_len=context["prev_len"],
                token_ids_delta=ids_delta,
                token_mask_delta=mask_delta,
                logprobs_delta=lp_delta,
            )
            key = staging_key(context["rollout_id"], context["call_id"])
            self.staged.append(key)  # durable before the response is released
            response["ng_commit_coords"] = {
                "rollout_id": context["rollout_id"],
                "call_id": context["call_id"],
                "parent_call_id": context["parent_call_id"],
                "delta_len": len(ids_delta),
                "cum_len": context["prev_len"] + len(ids_delta),
                "digest": digest,
                "staging_key": key,
                "weight_version": 4,
                "token_ids_delta": ids_delta,
            }
        return response


def _build_server(monkeypatch: MonkeyPatch, tmp_path, *, gate_enabled: bool = True):
    # Token capture (the base file-store layer) must be active for the
    # middleware to mint model_call_id and set the capture context -- the
    # #2124-c1 workaround: gate activation requires a capture dir.
    global_config = {
        "token_id_capture_enabled": True,
        "token_id_capture_dir": str(tmp_path / "token_capture"),
    }
    gate_config: dict[str, Any] = {"enabled": False}
    if gate_enabled:
        gate_config = {"enabled": True, "control_auth_token": "test-control-token"}
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8081,
        base_url="http://localhost:1/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy_model",
        entrypoint="",
        name="",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        token_capture_gate=gate_config,
    )
    monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", MagicMock(return_value=global_config))
    server = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict=global_config))
    worker = _FakeVLLMWorker()
    server._clients = [worker]
    return TestClient(server.setup_webserver()), worker


@pytest.fixture()
def gate_server(monkeypatch: MonkeyPatch, tmp_path) -> tuple[TestClient, _FakeVLLMWorker]:
    return _build_server(monkeypatch, tmp_path)


def _chat(client: TestClient, rollout_id: str, messages: list[dict], expect_status: int = 200) -> dict:
    response = client.post(f"/ng-rollout/{rollout_id}/v1/chat/completions", json={"messages": messages})
    assert response.status_code == expect_status, response.text
    return response.json()


def test_two_turn_conversation_prefix_and_receipt(gate_server) -> None:
    client, worker = gate_server
    assert client.put("/ng-control/rollouts/g7_r0", headers=AUTH).status_code == 200

    # Turn 1: nothing recorded -> text-mode root, full render.
    first = _chat(client, "g7_r0", [{"role": "user", "content": "hi"}])
    message = first["choices"][0]["message"]
    assert first["choices"][0].get("logprobs") is None, "logprobs must not reach the agent"
    assert "prompt_token_ids" not in message and "generation_token_ids" not in message
    assert "ng_commit_coords" not in first, "coords must not reach the agent"
    turn1_request = worker.requests[0]
    call1 = turn1_request["ng_capture"]["call_id"]
    assert turn1_request["ng_capture"]["mode"] == "text"
    assert "required_prefix_token_ids" not in turn1_request
    assert turn1_request["logprobs"] is True
    assert turn1_request["return_tokens_as_token_ids"] is True

    # Turn 2: the agent echoes history verbatim -> the assistant-turn
    # fingerprint resolves the parent -> token-in with exact ids.
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": message["content"]},
        {"role": "user", "content": "and then?"},
    ]
    _chat(client, "g7_r0", history)
    turn2_request = worker.requests[1]
    call2 = turn2_request["ng_capture"]["call_id"]
    assert turn2_request["ng_capture"]["mode"] == "token_in"
    assert turn2_request["ng_capture"]["parent_call_id"] == call1
    # Exact committed prefix: turn-1 render + turn-1 generation.
    assert turn2_request["required_prefix_token_ids"] == [10, 11, 12, 201, 202]
    assert turn2_request["ng_capture"]["prev_len"] == 5

    # Seal -> token-free receipt with the two-call chain.
    sealed = client.post("/ng-control/rollouts/g7_r0/seal", json={"reward": 1.0}, headers=AUTH)
    assert sealed.status_code == 200
    receipt = sealed.json()
    assert [entry["call_id"] for entry in receipt["manifest"]] == [call1, call2]
    assert [entry["parent_call_id"] for entry in receipt["manifest"]] == [None, call1]
    assert [entry["cum_len"] for entry in receipt["manifest"]] == [5, 9]  # 5 + (2 suffix + 2 generated)
    assert receipt["terminal_call_id"] == call2
    assert not receipt["capture_poisoned"]
    assert [entry["staging_key"] for entry in receipt["manifest"]] == worker.staged

    # Seal dropped the lineage: the index holds nothing for this rollout.
    metrics = client.get("/ng-control/metrics", headers=AUTH).json()
    assert metrics["token_in"] == 1
    assert metrics["sealed"] == 1
    assert metrics["lineage_rollouts"] == 0


def test_response_without_coords_poisons_but_still_serves(gate_server) -> None:
    client, worker = gate_server
    worker.attach_coords = False
    assert client.put("/ng-control/rollouts/r1", headers=AUTH).status_code == 200
    first = _chat(client, "r1", [{"role": "user", "content": "hi"}])
    assert first["choices"][0]["message"]["content"] == "turn 1", "completion still served"
    receipt = client.post("/ng-control/rollouts/r1/seal", json={"reward": 0.0}, headers=AUTH).json()
    assert receipt["capture_poisoned"] and receipt["manifest"] == []


def test_edited_assistant_history_falls_back_to_text_root(gate_server) -> None:
    client, worker = gate_server
    assert client.put("/ng-control/rollouts/r2", headers=AUTH).status_code == 200
    _chat(client, "r2", [{"role": "user", "content": "hi"}])
    # The harness rewrote the model's turn (compaction, truncation): the
    # fingerprint misses and the call falls back to a fresh root.
    edited_history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "REWRITTEN by the harness"},
        {"role": "user", "content": "next"},
    ]
    _chat(client, "r2", edited_history)
    assert worker.requests[1]["ng_capture"]["mode"] == "text"
    assert "required_prefix_token_ids" not in worker.requests[1]
    receipt = client.post("/ng-control/rollouts/r2/seal", json={"reward": 0.0}, headers=AUTH).json()
    # Two roots, both committed; correct but wasteful (§ 3.3).
    assert [entry["parent_call_id"] for entry in receipt["manifest"]] == [None, None]
    metrics = client.get("/ng-control/metrics", headers=AUTH).json()
    assert metrics["fallback_no_match"] == 1


def test_ambiguous_history_falls_back_and_is_counted(gate_server) -> None:
    client, worker = gate_server
    worker.fixed_content = "SAME TURN"
    assert client.put("/ng-control/rollouts/r3", headers=AUTH).status_code == 200
    # Two roots that served byte-identical assistant turns (a harness retry).
    _chat(client, "r3", [{"role": "user", "content": "hi"}])
    _chat(client, "r3", [{"role": "user", "content": "hi"}])
    # A child echoing that turn matches BOTH recorded calls: ambiguous, so the
    # gate must fall back rather than guess a parent.
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "SAME TURN"},
        {"role": "user", "content": "next"},
    ]
    _chat(client, "r3", history)
    assert worker.requests[2]["ng_capture"]["mode"] == "text"
    metrics = client.get("/ng-control/metrics", headers=AUTH).json()
    assert metrics["fallback_ambiguous"] == 1


def test_prefixed_but_unknown_rollout_is_rejected(gate_server) -> None:
    client, worker = gate_server
    response = client.post(
        "/ng-rollout/never-registered/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "hi"}]},
    )
    assert response.status_code == 409
    assert not worker.requests, "a rejected call must not reach the engine"


def test_uncorrelated_call_passes_through_untouched(gate_server) -> None:
    client, worker = gate_server
    response = client.post("/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]})
    assert response.status_code == 200
    assert "ng_capture" not in worker.requests[0], "uncorrelated traffic must not enter the gate"
    metrics = client.get("/ng-control/metrics", headers=AUTH).json()
    assert metrics["unattributed_calls"] == 1


def test_control_routes_require_bearer_token(gate_server) -> None:
    client, _ = gate_server
    assert client.put("/ng-control/rollouts/r9").status_code == 401
    assert client.put("/ng-control/rollouts/r9", headers={"Authorization": "Bearer wrong"}).status_code == 401
    assert client.post("/ng-control/rollouts/r9/seal", json={"reward": 0.0}).status_code == 401
    assert client.get("/ng-control/metrics").status_code == 401


def test_gate_disabled_leaves_legacy_path_untouched(monkeypatch: MonkeyPatch, tmp_path) -> None:
    client, worker = _build_server(monkeypatch, tmp_path, gate_enabled=False)
    # No control routes when dormant.
    assert client.put("/ng-control/rollouts/r0", headers=AUTH).status_code in (404, 405)
    # Correlated calls still flow through the base capture path unchanged.
    response = client.post(
        "/ng-rollout/r0/v1/chat/completions", json={"messages": [{"role": "user", "content": "hi"}]}
    )
    assert response.status_code == 200
    assert "ng_capture" not in worker.requests[0]


def test_gate_requires_control_auth_token(monkeypatch: MonkeyPatch, tmp_path) -> None:
    config = VLLMModelConfig(
        host="0.0.0.0",
        port=8081,
        base_url="http://localhost:1/v1",
        api_key="dummy_key",  # pragma: allowlist secret
        model="dummy_model",
        entrypoint="",
        name="",
        return_token_id_information=False,
        uses_reasoning_parser=False,
        token_capture_gate={"enabled": True},  # no control_auth_token
    )
    monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", MagicMock(return_value=dict()))
    server = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    with pytest.raises(Exception, match="auth token"):
        server.setup_webserver()


def test_gate_and_token_echo_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        VLLMModelConfig(
            host="0.0.0.0",
            port=8081,
            base_url="http://localhost:1/v1",
            api_key="dummy_key",  # pragma: allowlist secret
            model="dummy_model",
            entrypoint="",
            name="",
            return_token_id_information=True,
            uses_reasoning_parser=False,
            token_capture_gate={"enabled": True, "control_auth_token": "t"},
        )
