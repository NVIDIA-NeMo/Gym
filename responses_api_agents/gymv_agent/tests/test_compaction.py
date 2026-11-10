# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real adapter loops and dependency client; synthetic responses, no model GPUs."""

import json
from copy import deepcopy
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ServerDisconnectedError
from omegaconf import OmegaConf

from nemo_gym.agents.context_management import ContextGuardRejected, ContextHistoryConfig
from nemo_gym.client import ServerClient
from nemo_gym.core import NeMoGymResponseCreateParamsNonStreaming
from responses_api_agents.gymv_agent.app import GymVAgent as Agent
from responses_api_agents.gymv_agent.app import GymVAgentConfig as AgentConfig
from responses_api_agents.gymv_agent.app import GymVAgentRunRequest as RunRequest
from responses_api_agents.visual_context import visual_context_client


def history_config(enabled=True):
    return ContextHistoryConfig.model_validate(
        {
            "enabled": enabled,
            "policy": {
                "type": "recency",
                "config": {
                    "images": {"enabled": True, "protect_initial_context": True, "keep_last_groups": 2},
                    "reasoning": {"enabled": True, "keep_first_block": True, "keep_last_blocks": 2},
                },
            },
            "guards": {"max_total_tokens": 51200, "reserved_generation_tokens": 8192},
            "max_model_calls": 20,
        }
    )


def observation(index, feedback=True):
    content = [{"type": "input_image", "image_url": f"https://example.invalid/{index}.png", "detail": "auto"}]
    if feedback:
        content.append({"type": "input_text", "text": f"Observation {index}; rules and action history."})
    return {"role": "user", "type": "message", "content": content, "env_info": {"index": index}}


def response(index):
    return {
        "id": f"response-{index}",
        "created_at": 0,
        "model": "test",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "metadata": {"model_marker": "preserved"},
        "usage": {
            "input_tokens": 100,
            "output_tokens": 10,
            "total_tokens": 110,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens_details": {"reasoning_tokens": 4},
        },
        "output": [
            {
                "id": f"reason-{index}",
                "type": "reasoning",
                "summary": [],
                "content": [{"type": "reasoning_text", "text": f"Deductions {index}"}],
            },
            {
                "id": f"message-{index}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": rf"\boxed{{move{index}}}", "annotations": []}],
            },
        ],
    }


def http(payload):
    reply = MagicMock(ok=True, cookies={"session": "retained"})
    reply.json = AsyncMock(return_value=payload)
    reply.read = AsyncMock(return_value=json.dumps(payload).encode())
    return reply


class Episode:
    def __init__(self, *, enabled=True, turns=6, feedback=True, reject_at=None, failure_at=None, failure=None):
        self.calls, self.requests = [], []
        self.generated = self.measured = 0
        self.turns, self.feedback = turns, feedback
        self.reject_at, self.failure_at, self.failure = reject_at, failure_at, failure
        client = MagicMock(spec=ServerClient)
        client.global_config_dict = {"token_id_capture": {"enabled": True, "all_agents": True}}
        client.post = AsyncMock(side_effect=self.post)
        self.agent = Agent(
            config=AgentConfig(
                host="localhost",
                port=8080,
                entrypoint="app.py",
                name="visual-agent",
                model_server={"type": "responses_api_models", "name": "model"},
                resources_server={"type": "resources_servers", "name": "resources"},
                context_history=history_config(enabled) if enabled is not None else None,
                max_steps=turns,
            ),
            server_client=client,
        )
        # Canonical collated task alias must survive capture owner derivation.
        self.request = RunRequest.model_validate(
            {
                "task_idx": 0,
                "_ng_task_index": 7,
                "_ng_rollout_index": 2,
                "_ng_attempt_index": 1,
                "responses_create_params": {"input": [], "max_output_tokens": 8192},
            }
        )

    async def post(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        path = kwargs["url_path"]
        if path.endswith("/context/measure"):
            self.measured += 1
            return http({"prompt_token_count": 50000 if self.measured == self.reject_at else 100})
        if path.endswith("/v1/responses"):
            self.generated += 1
            if self.generated == self.failure_at:
                if self.failure == "invalid":
                    return http({"id": "malformed"})
                if self.failure == "json":
                    reply = http({})
                    reply.read.side_effect = json.JSONDecodeError("invalid", "", 0)
                    return reply
                raise ServerDisconnectedError("unacknowledged model call")
            self.requests.append(kwargs["json"].model_dump(mode="json"))
            return http(response(self.generated))
        if path == "/seed_session":
            return http({"env_id": "session-1", "obs": [observation(0, self.feedback)], "tools": []})
        if path == "/step":
            return http(
                {
                    "obs": [observation(self.generated, self.feedback)],
                    "reward": 0,
                    "done": self.generated == self.turns,
                }
            )
        if path == "/close":
            return http({"success": True, "message": "closed"})
        if path == "/verify":
            return http({**kwargs["json"], "reward": 0.5})
        raise AssertionError(path)

    async def run(self):
        return await self.agent.responses(self.request)


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [None, False, True])
@pytest.mark.parametrize("turns", [6, 20])
async def test_retention_capture_and_complete_returned_history(enabled, turns):
    episode = Episode(enabled=enabled, turns=turns)
    result = await episode.run()
    assert len(episode.requests) == turns
    final = episode.requests[-1]["input"]
    reasons = [item["id"] for item in final if item.get("type") == "reasoning"]
    assert reasons == (
        ["reason-1", f"reason-{turns - 2}", f"reason-{turns - 1}"]
        if enabled
        else [f"reason-{i}" for i in range(1, turns)]
    )
    images = [
        part["image_url"]
        for item in final
        for part in item.get("content", [])
        if isinstance(part, dict) and part.get("type") == "input_image"
    ]
    assert images == (
        [
            "https://example.invalid/0.png",
            f"https://example.invalid/{turns - 2}.png",
            f"https://example.invalid/{turns - 1}.png",
        ]
        if enabled
        else [f"https://example.invalid/{i}.png" for i in range(turns)]
    )
    assert [item["id"] for item in final if item.get("role") == "assistant"] == [
        f"message-{i}" for i in range(1, turns)
    ]
    assert [item.id for item in result.output if item.type == "reasoning"] == [
        f"reason-{i}" for i in range(1, turns + 1)
    ]
    assert result.id == f"response-{turns}"
    assert result.usage.total_tokens == 110 * turns
    assert result.usage.output_tokens_details.reasoning_tokens == 4 * turns
    assert result.metadata["model_marker"] == "preserved"
    assert result.group_id == "7"
    assert result.seed_obs[0].env_info == {"index": 0}
    assert all("env_info" not in item for item in final)
    paths = [c for c in episode.calls if c["url_path"].endswith(("/v1/responses", "/context/measure"))]
    assert len(paths) == turns * (1 if enabled is None else 2)
    assert all("/7-2-a1/training-token-capture/" in c["url_path"] for c in paths)
    if enabled is not None:
        assert all(c["_resend_after_disconnect"] is False for c in paths)
    # Terminal unseen screenshot must not be published as a captured prompt.
    assert result.output[-1].id == f"message-{turns}"


@pytest.mark.asyncio
@pytest.mark.parametrize("turns", [6, 20])
async def test_image_only_observations_keep_distinct_turn_boundaries(turns):
    episode = Episode(turns=turns, feedback=False)
    await episode.run()
    final = episode.requests[-1]["input"]
    messages = [item for item in final if item.get("role") in ("user", "assistant")]
    assert [item["role"] for item in messages] == [
        "user",
        *[role for _ in range(turns - 1) for role in ("assistant", "user")],
    ]
    assert all(item["content"] for item in messages)
    assert [item["id"] for item in messages if item["role"] == "assistant"] == [
        f"message-{i}" for i in range(1, turns)
    ]


@pytest.mark.asyncio
async def test_later_guard_keeps_only_real_accepted_terminal():
    episode = Episode(reject_at=3)
    result = await episode.run()
    assert episode.generated == 2
    assert result.id == "response-2"
    assert result.metadata["termination_reason"] == "context_length_exceeded"
    assert result.output[-1].id == "message-2"
    assert result.usage.total_tokens == 220
    assert [c for c in episode.calls if c["url_path"] == "/close"][-1]["json"].get("discard_reward") is not True


@pytest.mark.asyncio
async def test_first_guard_never_fabricates_or_scores_a_terminal():
    episode = Episode(reject_at=1)
    with pytest.raises(ContextGuardRejected):
        await episode.agent.run(episode.request)
    assert episode.generated == 0
    assert not any(c["url_path"] in ("/verify", "/step") for c in episode.calls)
    assert [c for c in episode.calls if c["url_path"] == "/close"][-1]["json"]["discard_reward"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["disconnect", "invalid", "json"])
async def test_ambiguous_or_malformed_response_never_returns_prior_success(failure):
    episode = Episode(failure_at=2, failure=failure)
    from pydantic import ValidationError

    with pytest.raises((ServerDisconnectedError, ValidationError, json.JSONDecodeError)):
        await episode.agent.run(episode.request)
    assert episode.generated == 2
    assert not any(c["url_path"] == "/verify" for c in episode.calls)
    assert [c for c in episode.calls if c["url_path"] == "/close"][-1]["json"]["discard_reward"]


@pytest.mark.parametrize(
    "invalid",
    [
        "capture",
        "owner",
        "reserve",
        "transitions",
        "retry",
        "context",
        "images",
        "budget_growth",
        "missing_limit",
        "missing_budget",
    ],
)
def test_admission_guards(invalid):
    agent = MagicMock()
    agent._token_id_capture_enabled.return_value = invalid != "capture"
    agent.config.return_transitions = invalid == "transitions"
    agent.config.max_no_boxed_truncation_retries = 1 if invalid == "budget_growth" else 0
    agent.config.no_boxed_truncation_retry_factor = 2
    agent.config.max_total_sequence_length = 100 if invalid == "context" else None
    agent.config.max_images = 5 if invalid == "images" else None
    cfg = history_config()
    if invalid == "retry":
        cfg.max_response_retries = 1
    if invalid == "reserve":
        cfg.guards.reserved_generation_tokens = 1
    if invalid == "missing_limit":
        cfg.guards.max_total_tokens = None
    with pytest.raises(ValueError):
        visual_context_client(
            agent,
            config=cfg,
            logical_rollout_id=None if invalid == "owner" else "owner-1",
            initial_request=NeMoGymResponseCreateParamsNonStreaming(
                input="task", max_output_tokens=None if invalid == "missing_budget" else 8192
            ),
        )


def test_shipped_overlay_and_guard_budget():
    path = Path(__file__).parents[1] / "configs/first_last_two.yaml"
    options = OmegaConf.to_container(OmegaConf.load(path), resolve=True)["gymv_agent"]["responses_api_agents"][
        "gymv_agent"
    ]
    cfg = AgentConfig(
        host="localhost",
        port=8080,
        name="agent",
        entrypoint="app.py",
        resources_server={"type": "resources_servers", "name": "resources"},
        model_server={"type": "responses_api_models", "name": "model"},
        **options,
    )
    assert cfg.context_history == history_config()
    assert cfg.max_steps == cfg.context_history.max_model_calls == 20


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_path", [None, "/context/measure", "/v1/responses"])
async def test_real_http_capture_cookie_and_no_ambiguous_replay(drop_path):
    import asyncio
    from collections import Counter

    from aiohttp import web

    from nemo_gym.client import get_global_aiohttp_client
    from nemo_gym.config import BaseServerConfig

    episode = Episode()
    counts = Counter()
    prefix = "/ng-rollout/7-2-a1/training-token-capture"

    async def handle(request):
        payload = await request.json()
        counts[request.path] += 1
        if request.path in (prefix + "/context/measure", prefix + "/v1/responses"):
            if counts[request.path] > 1:
                assert request.cookies.get("session") == "retained"
            if drop_path and request.path == prefix + drop_path and counts[request.path] == 2:
                request.transport.close()
                return web.Response()
            payload = NeMoGymResponseCreateParamsNonStreaming.model_validate(payload)
        reply = await episode.post(url_path=request.path, json=payload)
        result = web.json_response(await reply.json())
        if request.path.endswith("/v1/responses"):
            result.set_cookie("session", "retained")
        return result

    app = web.Application()
    app.router.add_post("/{path:.*}", handle)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    episode.agent.server_client = ServerClient(
        head_server_config=BaseServerConfig(host="127.0.0.1", port=port),
        global_config_dict=OmegaConf.create(
            {
                "token_id_capture": {"enabled": True, "all_agents": True},
                "model": {"responses_api_models": {"test": {"host": "127.0.0.1", "port": port}}},
                "resources": {"resources_servers": {"test": {"host": "127.0.0.1", "port": port}}},
            }
        ),
    )
    try:
        async with asyncio.timeout(10):
            if drop_path:
                with pytest.raises(ServerDisconnectedError):
                    await episode.agent.run(episode.request)
                assert counts[prefix + drop_path] == 2
                assert counts["/verify"] == 0
                assert [c for c in episode.calls if c["url_path"] == "/close"][-1]["json"]["discard_reward"]
            else:
                result = await episode.agent.run(episode.request)
                assert result.reward == 0.5
                assert result.response.id == "response-6"
                assert result.response.usage.total_tokens == 660
                assert counts[prefix + "/v1/responses"] == counts[prefix + "/context/measure"] == 6
                assert counts["/verify"] == 1
    finally:
        await get_global_aiohttp_client().close()
        await runner.cleanup()
