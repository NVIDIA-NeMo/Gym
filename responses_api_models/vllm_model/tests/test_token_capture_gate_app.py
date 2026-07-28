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
"""S3: gate-mode VLLMModel end-to-end over HTTP (fake vLLM worker).

Drives the model server exactly as the RL controller + agent + worker would:
register over the control route, run a two-turn chat conversation where the
fake worker honors ``required_prefix_token_ids`` and returns ``CommitCoords``,
then seal and check the receipt. Asserts the wire properties the design
promises: exact prefix service, marker attach (and no token arrays / logprobs
on the agent-facing response), coords ingestion as the commit, and
capture-failed poisoning when a response carries no coords.
"""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient
from pytest import MonkeyPatch

import nemo_gym.server_utils
from nemo_gym.server_utils import ServerClient
from nemo_gym.token_id_capture.staging.digest import build_staging_delta, compute_staging_digest
from nemo_gym.token_id_capture.staging.records import staging_key
from responses_api_models.vllm_model.app import VLLMModel, VLLMModelConfig


class _FakeVLLMWorker:
    """A capture-enabled worker: splices the served prefix, 'generates' fixed
    ids, stages the delta (digest recorded), and rides coords on the response."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.staged: list[str] = []
        self.attach_coords = True
        self._turn = 0

    async def create_chat_completion(self, **body: Any) -> dict:
        self.requests.append(body)
        context = body["ng_capture"]
        prefix = body.get("required_prefix_token_ids") or []
        self._turn += 1
        if context["mode"] == "token_in":
            prompt_ids = list(prefix) + [100 + self._turn, 101 + self._turn]  # spliced suffix render
        else:
            prompt_ids = [10, 11, 12]  # full template render
        generated = [200 + self._turn, 201 + self._turn]
        logprobs = [-0.25, -0.5]

        response: dict = {
            "id": "chtcmpl-1",
            "object": "chat.completion",
            "created": 0,
            "model": body.get("model", "dummy"),
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": f"turn {self._turn}"},
                    # What return_tokens_as_token_ids puts on the wire; the
                    # gate must strip it before the agent sees the response.
                    "logprobs": {
                        "content": [{"token": f"token_id:{t}", "logprob": lp} for t, lp in zip(generated, logprobs)]
                    },
                }
            ],
        }
        if self.attach_coords:
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


@pytest.fixture()
def gate_server(monkeypatch: MonkeyPatch) -> tuple[TestClient, _FakeVLLMWorker]:
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
        token_capture_gate={"enabled": True},
    )
    monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", MagicMock(return_value=dict()))
    server = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    worker = _FakeVLLMWorker()
    server._clients = [worker]
    return TestClient(server.setup_webserver()), worker


def _chat(client: TestClient, rollout_id: str, messages: list[dict]) -> dict:
    response = client.post(
        "/v1/chat/completions",
        json={"messages": messages, "metadata": {"ng_rollout_id": rollout_id}},
    )
    assert response.status_code == 200, response.text
    return response.json()


def test_two_turn_conversation_prefix_marker_receipt(gate_server) -> None:
    client, worker = gate_server
    assert client.put("/ng-control/rollouts/g7_r0").status_code == 200

    # Turn 1: no marker -> text-mode root, full render.
    first = _chat(client, "g7_r0", [{"role": "user", "content": "hi"}])
    message = first["choices"][0]["message"]
    marker = message.get("ng_call_id")
    assert marker, "committed call must release a marker"
    assert first["choices"][0].get("logprobs") is None, "logprobs must not reach the agent"
    assert "prompt_token_ids" not in message and "generation_token_ids" not in message
    assert worker.requests[0]["ng_capture"]["mode"] == "text"
    assert "required_prefix_token_ids" not in worker.requests[0]
    assert worker.requests[0]["logprobs"] is True
    assert worker.requests[0]["return_tokens_as_token_ids"] is True

    # Turn 2: the agent echoes history verbatim -> token-in with exact ids.
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": message["content"], "ng_call_id": marker},
        {"role": "user", "content": "and then?"},
    ]
    second = _chat(client, "g7_r0", history)
    turn2_request = worker.requests[1]
    assert turn2_request["ng_capture"]["mode"] == "token_in"
    assert turn2_request["ng_capture"]["parent_call_id"] == marker
    # Exact committed prefix: turn-1 render + turn-1 generation.
    assert turn2_request["required_prefix_token_ids"] == [10, 11, 12, 201, 202]
    assert turn2_request["ng_capture"]["prev_len"] == 5
    marker2 = second["choices"][0]["message"]["ng_call_id"]

    # Seal -> token-free receipt with the two-call chain.
    sealed = client.post("/ng-control/rollouts/g7_r0/seal", json={"reward": 1.0})
    assert sealed.status_code == 200
    receipt = sealed.json()
    assert [entry["call_id"] for entry in receipt["manifest"]] == [marker, marker2]
    assert [entry["parent_call_id"] for entry in receipt["manifest"]] == [None, marker]
    assert [entry["cum_len"] for entry in receipt["manifest"]] == [5, 9]  # 5 + (2 suffix + 2 generated)
    assert receipt["terminal_call_id"] == marker2
    assert not receipt["capture_poisoned"]
    assert [entry["staging_key"] for entry in receipt["manifest"]] == worker.staged


def test_response_without_coords_poisons_but_still_serves(gate_server) -> None:
    client, worker = gate_server
    worker.attach_coords = False
    assert client.put("/ng-control/rollouts/r1").status_code == 200
    first = _chat(client, "r1", [{"role": "user", "content": "hi"}])
    assert "ng_call_id" not in first["choices"][0]["message"], "no marker without a commit"
    receipt = client.post("/ng-control/rollouts/r1/seal", json={"reward": 0.0}).json()
    assert receipt["capture_poisoned"] and receipt["manifest"] == []


def test_edited_history_falls_back_to_text_root(gate_server) -> None:
    client, worker = gate_server
    assert client.put("/ng-control/rollouts/r2").status_code == 200
    first = _chat(client, "r2", [{"role": "user", "content": "hi"}])
    marker = first["choices"][0]["message"]["ng_call_id"]
    edited_history = [
        {"role": "user", "content": "hi, REWRITTEN"},
        {"role": "assistant", "content": first["choices"][0]["message"]["content"], "ng_call_id": marker},
        {"role": "user", "content": "next"},
    ]
    _chat(client, "r2", edited_history)
    assert worker.requests[1]["ng_capture"]["mode"] == "text"
    assert "required_prefix_token_ids" not in worker.requests[1]
    receipt = client.post("/ng-control/rollouts/r2/seal", json={"reward": 0.0}).json()
    # Two roots, both committed; correct but wasteful (§ 3.3).
    assert [entry["parent_call_id"] for entry in receipt["manifest"]] == [None, None]


def test_gate_disabled_leaves_legacy_path_untouched(monkeypatch: MonkeyPatch) -> None:
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
    )
    monkeypatch.setattr(nemo_gym.server_utils, "get_global_config_dict", MagicMock(return_value=dict()))
    server = VLLMModel(config=config, server_client=MagicMock(spec=ServerClient, global_config_dict={}))
    assert server._gate is None
    client = TestClient(server.setup_webserver())
    # No control routes when dormant.
    assert client.put("/ng-control/rollouts/r0").status_code in (404, 405)


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
            token_capture_gate={"enabled": True},
        )
