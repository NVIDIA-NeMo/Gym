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
"""The repair loop with the model and resources servers replaced by a scripted client."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from app import VeriSoftBenchAgent, VeriSoftBenchAgentConfig, VeriSoftBenchAgentRunRequest, repair_params
from fastapi import Request

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient


def _params(**extra) -> NeMoGymResponseCreateParamsNonStreaming:
    return NeMoGymResponseCreateParamsNonStreaming(
        input=[{"role": "system", "content": "SYS"}, {"role": "user", "content": "prove"}],
        temperature=0.7,
        max_output_tokens=8192,
        **extra,
    )


def _model_json(text: str) -> dict[str, Any]:
    return {
        "id": "resp",
        "created_at": 0.0,
        "model": "m",
        "object": "response",
        "output": [
            {
                "id": "msg",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class FakeHTTPResponse:
    ok = True
    status = 200
    cookies: dict = {}

    def __init__(self, payload):
        self._payload = payload

    async def json(self):
        return self._payload


class ScriptedClient:
    """Answers ``/v1/responses`` with canned text and ``/verify`` from a script of statuses."""

    def __init__(self, verdicts: list[dict[str, Any]]):
        self.verdicts = list(verdicts)
        self.posts: list[tuple[str, str, Any]] = []

    async def post(self, server_name, url_path, json=None, cookies=None):
        self.posts.append((server_name, url_path, json))
        if url_path == "/seed_session":
            return FakeHTTPResponse({})
        if url_path.endswith("/v1/responses"):
            return FakeHTTPResponse(_model_json(f"<lean4_proof>attempt {len(self.posts)}</lean4_proof>"))
        assert url_path == "/verify"
        verdict = dict(self.verdicts.pop(0))
        verdict.setdefault("responses_create_params", json["responses_create_params"])
        verdict.setdefault("response", json["response"])
        verdict["repair_round"] = json["repair_round"]
        return FakeHTTPResponse(verdict)


def _agent(client) -> VeriSoftBenchAgent:
    config = VeriSoftBenchAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="verisoftbench_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="verisoftbench"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        max_repair_rounds=3,
    )
    server_client = MagicMock(spec=ServerClient)
    if hasattr(client, "post"):
        server_client.post = client.post
    return VeriSoftBenchAgent(config=config, server_client=server_client)


def _fail(round_index: int) -> dict[str, Any]:
    return {
        "reward": 0.0,
        "status": "lean_error",
        "needs_repair": True,
        "repair_prompt": f"FIX round {round_index}",
        "repair_state": {"proof": f"p{round_index}", "aux_lemmas": "", "err_msg": "e"},
        "harness_failure": 0.0,
    }


def _ok() -> dict[str, Any]:
    return {"reward": 1.0, "status": "verified", "needs_repair": False, "harness_failure": 0.0}


@pytest.fixture(autouse=True)
def plain_run_paths(monkeypatch):
    # No rollout-capture prefix in these tests; the scripted client matches bare paths.
    monkeypatch.setattr(VeriSoftBenchAgent, "url_path_for_run", lambda self, path, body: path)


async def _run(client) -> Any:
    agent = _agent(client)
    body = VeriSoftBenchAgentRunRequest(responses_create_params=_params(), verifier_metadata={"id": 1})
    request = MagicMock(spec=Request)
    request.cookies = {}
    return await agent.run(request, body)


class TestRepairParams:
    def test_keeps_sampling_and_system_turn_replaces_user_turn(self) -> None:
        new = repair_params(_params(), "FIX ME")
        assert new.temperature == 0.7 and new.max_output_tokens == 8192
        roles = [m.role for m in new.input]
        assert roles == ["system", "user"]
        assert new.input[0].content == "SYS" and new.input[1].content == "FIX ME"

    def test_string_input_has_no_system_turn(self) -> None:
        new = repair_params(NeMoGymResponseCreateParamsNonStreaming(input="prove"), "FIX ME")
        assert [m.role for m in new.input] == ["user"]


class TestLoop:
    @pytest.mark.asyncio
    async def test_bounded_at_three_repairs_and_records_every_round(self) -> None:
        client = ScriptedClient(
            [_fail(0), _fail(1), _fail(2), {**_fail(3), "needs_repair": False, "repair_prompt": None}]
        )
        final = await _run(client)
        verifies = [p for p in client.posts if p[1] == "/verify"]
        assert len(verifies) == 4 and [p[2]["repair_round"] for p in verifies] == [0, 1, 2, 3]
        assert final.reward == 0.0 and final.repair_rounds_used == 3 and final.total_rounds == 4
        assert final.solved_without_repair == 0.0 and len(final.attempts) == 4
        # Each repair round forwards the previous round's state and sends upstream's repair prompt.
        assert verifies[1][2]["repair_state"] == {"proof": "p0", "aux_lemmas": "", "err_msg": "e"}
        assert verifies[3][2]["repair_state"]["proof"] == "p2"
        gens = [p for p in client.posts if p[1].endswith("/v1/responses")]
        second_input = gens[1][2].input
        assert second_input[0].role == "system" and second_input[1].content == "FIX round 0"
        assert gens[1][2].temperature == 0.7

    @pytest.mark.asyncio
    async def test_agent_refuses_a_fourth_repair_even_if_the_server_offers_one(self) -> None:
        client = ScriptedClient([_fail(0), _fail(1), _fail(2), _fail(3), _ok()])
        final = await _run(client)
        assert len([p for p in client.posts if p[1] == "/verify"]) == 4 and final.reward == 0.0

    @pytest.mark.asyncio
    async def test_stops_at_first_verified_round(self) -> None:
        client = ScriptedClient([_fail(0), _ok()])
        final = await _run(client)
        assert final.reward == 1.0 and final.repair_rounds_used == 1 and final.solved_without_repair == 0.0
        assert len([p for p in client.posts if p[1] == "/verify"]) == 2

    @pytest.mark.asyncio
    async def test_solved_without_repair(self) -> None:
        final = await _run(ScriptedClient([_ok()]))
        assert final.reward == 1.0 and final.solved_without_repair == 1.0 and final.repair_rounds_used == 0

    @pytest.mark.asyncio
    async def test_harness_failure_ends_the_rollout(self) -> None:
        client = ScriptedClient([{**_fail(0), "harness_failure": 1.0, "status": "backend_error", "mask_sample": True}])
        final = await _run(client)
        assert final.reward == 0.0 and final.total_rounds == 1 and final.mask_sample is True

    @pytest.mark.asyncio
    async def test_missing_repair_prompt_ends_the_rollout(self) -> None:
        client = ScriptedClient([{**_fail(0), "repair_prompt": None}])
        final = await _run(client)
        assert final.total_rounds == 1


class TestSanity:
    def test_construct(self) -> None:
        _agent(MagicMock(spec=ServerClient))
