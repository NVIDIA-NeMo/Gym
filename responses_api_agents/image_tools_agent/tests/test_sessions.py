# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The image tools agent through the agent sessions API and the tool reward server."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import orjson
import pytest
from aiohttp import ClientResponseError
from fastapi.testclient import TestClient
from omegaconf import OmegaConf
from PIL import Image

from environment_servers.single_agent_turn.app import (
    SingleAgentTurnEnvironmentServer,
    SingleAgentTurnEnvironmentServerConfig,
)
from nemo_gym.agents import AGENT_ACTIVATION_HEADER, AgentSeedSessionRequest, check_agent_seed_rejected
from nemo_gym.client import ServerClient
from nemo_gym.config import AgentServerRef, BaseServerConfig, ResourcesServerRef
from nemo_gym.core import EpisodeId, MaterializedTask, TaskId
from nemo_gym.sandbox import DirectSandboxConnection, SandboxAccess
from nemo_gym.server_runtime import DirectHTTPToolAccess, SingleAgentTurnRequest, SingleAgentTurnTaskInput
from resources_servers.image_tools.base import IMAGE_TOOLS_ROLLOUT_KEY
from resources_servers.image_tools.tool_reward_app import ImageToolsRewardConfig, ImageToolsRewardServer
from responses_api_agents.image_tools_agent.app import (
    ImageToolsAgent,
    ImageToolsAgentConfig,
    ImageToolsAgentRunRequest,
)


ZOOM = (
    "<tool_call><function=image_zoom_in_tool>"
    "<parameter=bbox_2d>[0, 0, 500, 500]</parameter>"
    "<parameter=label>upper left</parameter>"
    "<parameter=img_idx>0</parameter>"
    "</function></tool_call>"
)
BASE_REF = {"type": "responses_api_agents", "name": "string_match_simple_agent"}


def model_reply(text: str) -> dict[str, Any]:
    return {
        "id": f"resp-{len(text)}",
        "created_at": 0.0,
        "model": "policy",
        "object": "response",
        "output": [
            {
                "type": "message",
                "role": "assistant",
                "id": f"msg-{len(text)}",
                "status": "completed",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
    }


class _Reply:
    """A server reply in the shape ServerClient callers read."""

    def __init__(self, body: bytes, *, status: int = 200, cookies: dict[str, str] | None = None, headers=None):
        self.status = status
        self.ok = status < 400
        self.headers = headers or {}
        self.cookies = {name: SimpleNamespace(value=value) for name, value in (cookies or {}).items()}
        self._body = body
        self.content = SimpleNamespace(read=self.read)

    async def read(self) -> bytes:
        return self._body

    def raise_for_status(self) -> None:
        if not self.ok:
            raise ClientResponseError(MagicMock(), (), status=self.status, message="error")


class _Transport(ServerClient):
    """Scripted model and base verifier; every other server is an app served in process."""

    model_config = {"arbitrary_types_allowed": True}
    apps: dict = {}
    model_turns: list = []
    model_calls: int = 0
    base_reward: float = 0.5

    async def post(self, server_name: str, url_path: str, json=None, cookies=None, headers=None, **kwargs):
        body = json.model_dump(mode="json", exclude_unset=True) if hasattr(json, "model_dump") else json
        if server_name == "policy_model":
            self.model_calls += 1
            return _Reply(orjson.dumps(model_reply(self.model_turns[(self.model_calls - 1) % len(self.model_turns)])))
        if server_name == "string_match":
            if url_path == "/seed_session":
                return _Reply(b"{}")
            # The base verifier keeps no state; it scores the final answer it is sent.
            assert len(body["response"]["output"]) == 1
            return _Reply(orjson.dumps(body | {"reward": self.base_reward}))
        client = self.apps[server_name]
        client.cookies.clear()
        client.cookies.update(dict(cookies or {}))
        reply = client.post(url_path, json=body, headers=headers)
        return _Reply(reply.content, status=reply.status_code, cookies=dict(reply.cookies), headers=reply.headers)


AGENT_SETTINGS = {
    "max_steps": 5,
    "tool_success_reward": 0.02,
    "tool_success_reward_cap": 0.05,
    "resource_servers_by_agent": {"string_match_simple_agent": {"type": "resources_servers", "name": "string_match"}},
}


def make_transport(tmp_path: Path, base_reward: float = 0.5) -> _Transport:
    agent_config = AGENT_SETTINGS | {"crop_dir": str(tmp_path / "crops"), "host": "agent", "port": 8080}
    return _Transport(
        head_server_config=BaseServerConfig(host="head", port=1),
        global_config_dict=OmegaConf.create(
            {
                "image_tools_simple_agent": {"responses_api_agents": {"image_tools_agent": agent_config}},
                "image_tools_reward": {"resources_servers": {"image_tools": {"host": "reward", "port": 9000}}},
            }
        ),
        apps={},
        model_turns=[ZOOM, "The answer is car."],
        base_reward=base_reward,
    )


def make_agent(transport: ServerClient, tmp_path: Path) -> ImageToolsAgent:
    return ImageToolsAgent(
        config=ImageToolsAgentConfig(
            host="agent",
            port=8080,
            entrypoint="app.py",
            name="image_tools_simple_agent",
            model_server={"type": "responses_api_models", "name": "policy_model"},
            crop_dir=str(tmp_path / "crops"),
            **{k: v for k, v in AGENT_SETTINGS.items() if k != "resource_servers_by_agent"},
            resource_servers_by_agent={
                name: ResourcesServerRef(**ref) for name, ref in AGENT_SETTINGS["resource_servers_by_agent"].items()
            },
        ),
        server_client=transport,
    )


@pytest.fixture
def row(tmp_path: Path) -> dict[str, Any]:
    image = tmp_path / "source.png"
    Image.new("RGB", (256, 192), color=(120, 80, 40)).save(image)
    return {
        "responses_create_params": {
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What object is shown?"},
                        {"type": "input_image", "image_url": str(image), "detail": "auto"},
                    ],
                }
            ]
        },
        "image_tools_base_agent_ref": BASE_REF,
        "expected_answer": "car",
    }


def seed_request(**overrides) -> AgentSeedSessionRequest:
    fields = {
        "agent_session_id": "agent-session",
        "episode_id": EpisodeId(rollout_id="rollout", attempt=1),
        "task_id": TaskId(taskset="image_tools", task_id="task"),
    }
    return AgentSeedSessionRequest(**(fields | overrides))


def test_a_session_activation_runs_the_tool_loop_once(tmp_path: Path, row: dict[str, Any]) -> None:
    transport = make_transport(tmp_path)
    client = TestClient(make_agent(transport, tmp_path).setup_webserver())
    assert client.post("/v1/agent_sessions", json=seed_request().model_dump(mode="json")).status_code == 200

    params = row["responses_create_params"]
    reply = client.post("/v1/responses", json=params, headers={AGENT_ACTIVATION_HEADER: "1"})

    assert reply.status_code == 200, reply.text
    body = reply.json()
    assert [item["role"] for item in body["output"]] == ["assistant", "user", "assistant"]
    crops = body[IMAGE_TOOLS_ROLLOUT_KEY]["image_tools_output_paths"]
    assert len(crops) == 1 and Path(crops[0]).exists()
    assert body[IMAGE_TOOLS_ROLLOUT_KEY]["image_tools_generation_image_paths"] == [[], crops]

    # A retried activation gets the rollout it already ran.
    assert client.post("/v1/responses", json=params, headers={AGENT_ACTIVATION_HEADER: "1"}).json() == body
    assert transport.model_calls == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "sandbox_access": SandboxAccess(
                connection=DirectSandboxConnection(provider_config_ref="docker", descriptor={}), workdir="/w"
            )
        },
        {"tool_accesses": [DirectHTTPToolAccess(name="tools", required=True, base_url="http://tools.test")]},
    ],
)
def test_a_seed_with_a_sandbox_or_required_tools_is_refused(tmp_path: Path, overrides: dict[str, Any]) -> None:
    app = make_agent(make_transport(tmp_path), tmp_path).setup_webserver()
    assert "does not support" in check_agent_seed_rejected(app, seed_request(**overrides))


@pytest.mark.parametrize("base_reward", [0.5, 1.0])
async def test_an_environment_server_episode_scores_as_the_agents_own_run(
    tmp_path: Path, row: dict[str, Any], base_reward: float
) -> None:
    transport = make_transport(tmp_path, base_reward)
    agent = make_agent(transport, tmp_path)
    reward_server = ImageToolsRewardServer(
        config=ImageToolsRewardConfig(
            host="reward",
            port=9000,
            entrypoint="tool_reward_app.py",
            name="image_tools_reward",
            agent_server=AgentServerRef(type="responses_api_agents", name="image_tools_simple_agent"),
        ),
        server_client=transport,
    )
    transport.apps.update(
        image_tools_reward=TestClient(reward_server.setup_webserver()),
        image_tools_simple_agent=TestClient(agent.setup_webserver()),
    )
    environment = SingleAgentTurnEnvironmentServer(
        config=SingleAgentTurnEnvironmentServerConfig(
            name="environment",
            host="environment",
            port=8002,
            entrypoint="app.py",
            resources_server=ResourcesServerRef(type="resources_servers", name="image_tools_reward"),
            agent_server=AgentServerRef(type="responses_api_agents", name="image_tools_simple_agent"),
            cleanup_timeout_seconds=10,
        ),
        server_client=transport,
    )

    episode = await environment.run_request(
        SingleAgentTurnRequest(
            episode_id=EpisodeId(rollout_id="rollout", attempt=1),
            task=MaterializedTask(
                task_id=TaskId(taskset="image_tools", task_id="task"),
                task_input=SingleAgentTurnTaskInput.model_validate(row),
            ),
        )
    )
    legacy = (await agent.run(SimpleNamespace(cookies={}), ImageToolsAgentRunRequest.model_validate(row))).model_dump()

    assert episode.failure is None, episode.failure
    result = episode.result.model_dump()
    for field in (
        "reward",
        "base_reward",
        "image_tools_aux_reward",
        "image_tools_call_count",
        "image_tools_error_count",
    ):
        assert result[field] == pytest.approx(legacy[field]), field
    assert result["reward"] == pytest.approx(min(1.0, base_reward + 0.02))
    assert len(result["image_tools_output_paths"]) == 1
    assert result["image_tools_base_agent_ref"]["name"] == "string_match_simple_agent"
    assert [item["role"] for item in result["response"]["output"]] == ["assistant", "user", "assistant"]
