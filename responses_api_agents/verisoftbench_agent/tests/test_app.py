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

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import orjson
import pytest
from app import VeriSoftBenchAgent, VeriSoftBenchAgentConfig, VeriSoftBenchAgentRunRequest, repair_params
from fastapi import Request
from fastapi.testclient import TestClient

from nemo_gym.base_resources_server import AggregateMetricsRequest
from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.global_config import ROLLOUT_INDEX_KEY_NAME, TASK_INDEX_KEY_NAME
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

    async def read(self):
        return orjson.dumps(self._payload)


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
    monkeypatch.setattr(VeriSoftBenchAgent, "url_path_for_request", lambda self, path, request: path)


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


class TestResponses:
    @pytest.mark.asyncio
    async def test_responses_forwards_to_the_policy_model_and_wraps_string_input(self) -> None:
        client = ScriptedClient([])
        agent = _agent(client)
        request = MagicMock(spec=Request)
        request.cookies = {"sid": "1"}
        response = MagicMock()
        out = await agent.responses(request, response, NeMoGymResponseCreateParamsNonStreaming(input="prove"))
        server_name, path, sent = client.posts[0]
        assert server_name == "policy_model" and path.endswith("/v1/responses")
        assert [m.role for m in sent.input] == ["user"]
        assert out.output[0].content[0].text.startswith("<lean4_proof>")


def _real_resources_server():
    """The real verisoftbench resources server, harness-free.

    Its flat ``upstream`` import resolves only with the server directory on sys.path, and
    ``/nonexistent`` makes harness setup fail closed before any git or docker access.
    """
    server_dir = Path(__file__).resolve().parents[3] / "resources_servers" / "verisoftbench"
    sys.path.insert(0, str(server_dir))
    try:
        from resources_servers.verisoftbench.app import (
            VeriSoftBenchResourcesServer,
            VeriSoftBenchResourcesServerConfig,
        )
    finally:
        sys.path.remove(str(server_dir))
    config = VeriSoftBenchResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="verisoftbench",
        upstream_dir="/nonexistent",
        require_container_at_startup=False,
    )
    return VeriSoftBenchResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


class AggregatingClient:
    """Routes the agent's ``/aggregate_metrics`` forward to a real resources server."""

    def __init__(self, server):
        self.server = server
        self.posts: list[tuple[str, str, AggregateMetricsRequest]] = []

    async def post(self, server_name, url_path, json=None, cookies=None):
        assert url_path == "/aggregate_metrics"
        request = json if isinstance(json, AggregateMetricsRequest) else AggregateMetricsRequest.model_validate(json)
        self.posts.append((server_name, url_path, request))
        metrics = await self.server.aggregate_metrics(request)
        return FakeHTTPResponse(metrics.model_dump())


def _row(task: int, rollout: int, reward: float, without_repair: float, category: str, **extra) -> dict[str, Any]:
    return {
        TASK_INDEX_KEY_NAME: task,
        ROLLOUT_INDEX_KEY_NAME: rollout,
        "reward": reward,
        "solved_without_repair": without_repair,
        "repair_rounds_used": 1 if reward >= 1.0 and without_repair < 1.0 else 0,
        "harness_failure": 0.0,
        "category": category,
        **extra,
    }


class TestAggregateMetrics:
    def test_proxies_to_the_resources_server_and_reports_the_any_of_k_headline(self) -> None:
        """Gym aggregates through the agent's /aggregate_metrics; prove that boundary forwards.

        Two tasks: task 0 is solved on repair only, task 1 is unsolved — its only
        verified row is masked, so counting it would flip ``any_of_k/solved`` to 100.
        """
        rows = [
            _row(0, 0, 1.0, 0.0, "Compiler"),
            _row(0, 1, 0.0, 0.0, "Compiler"),
            _row(1, 0, 0.0, 0.0, "Semantics"),
            _row(1, 1, 1.0, 1.0, "Semantics", mask_sample=True, harness_failure=1.0),
        ]
        client = AggregatingClient(_real_resources_server())
        agent = _agent(client)
        agent.server_client.global_config_dict = {"observability_enabled": True}
        http = TestClient(agent.setup_webserver())

        result = http.post("/aggregate_metrics", json={"verify_responses": rows})

        assert result.status_code == 200
        server_name, _, forwarded = client.posts[0]
        assert server_name == "verisoftbench" and forwarded.verify_responses == rows
        am = result.json()["agent_metrics"]
        assert am["any_of_k/solved"] == 50.0
        assert am["any_of_k/solved_without_repair"] == 0.0
        assert am["any_of_k/solved/compiler"] == 100.0 and am["any_of_k/solved/semantics"] == 0.0
        assert am["count/tasks"] == 2
        # The masked row leaves the quality set the way Gym's partition does, and is
        # reported as coverage instead.
        assert am["coverage/masked_rollouts"] == 1 and am["coverage/measured_rollouts"] == 3
        key_metrics = result.json()["key_metrics"]
        assert key_metrics["any_of_k/solved"] == 50.0 and key_metrics["any_of_k/solved_without_repair"] == 0.0
        # Headline selection is the server's: per-category breakdowns stay out.
        assert "any_of_k/solved/compiler" not in key_metrics
