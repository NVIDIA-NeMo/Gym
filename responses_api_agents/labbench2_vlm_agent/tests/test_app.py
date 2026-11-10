# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from nemo_gym.agents import AgentCloseSessionRequest, AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.config import ModelServerRef, ResourcesServerRef
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.server_runtime import AgentContext, SingleAgentTurnTaskInput
from responses_api_agents.labbench2_vlm_agent import app as labbench_agent_app
from responses_api_agents.labbench2_vlm_agent.app import (
    LabbenchVLMAgent,
    LabbenchVLMAgentConfig,
    _effective_media_mode,
    _strip_image_blocks,
)
from responses_api_agents.simple_agent.app import SimpleAgent, SimpleAgentRunRequest, SimpleAgentVerifyResponse


def _agent_config(**overrides) -> LabbenchVLMAgentConfig:
    config_kwargs = {
        "name": "labbench2_vlm_simple_agent",
        "host": "0.0.0.0",
        "port": 8080,
        "entrypoint": "app.py",
        "resources_server": ResourcesServerRef(type="resources_servers", name="labbench2_vlm"),
        "model_server": ModelServerRef(type="responses_api_models", name="policy_model"),
        "media_base_dir": "resources_servers/labbench2_vlm/data",
        "dpi": 170,
    }
    config_kwargs.update(overrides)
    return LabbenchVLMAgentConfig(**config_kwargs)


def test_agent_instantiation() -> None:
    config = _agent_config()
    agent = LabbenchVLMAgent(config=config, server_client=MagicMock(spec=ServerClient))
    assert agent.config.media_base_dir == "resources_servers/labbench2_vlm/data"
    assert agent.config.dpi == 170


_PNG = bytes.fromhex("89504e470d0a1a0a0000000d4948445200000001000000010806000000")


def _model_reply(response_id: str) -> MagicMock:
    payload = {
        "id": response_id,
        "created_at": 1.0,
        "model": "policy",
        "object": "response",
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "output": [
            {
                "id": f"msg-{response_id}",
                "type": "message",
                "role": "assistant",
                "status": "completed",
                "content": [{"type": "output_text", "text": "A cat.", "annotations": []}],
            }
        ],
    }
    reply = MagicMock(status=200, cookies={}, ok=True)
    reply.read = AsyncMock(return_value=json.dumps(payload))
    return reply


def _seed(media_path: str | None) -> AgentSeedSessionRequest:
    return AgentSeedSessionRequest(
        agent_session_id="labbench2-session",
        episode_id=EpisodeId(rollout_id="rollout", attempt=0),
        task_id=TaskId(taskset="labbench2_vlm", task_id="0"),
        agent_context=(
            AgentContext(values={"tag": "figqa2-img"}, artifacts={"media": {"path": media_path}})
            if media_path is not None
            else None
        ),
    )


@pytest.fixture
def session_agent(tmp_path: Path) -> tuple[LabbenchVLMAgent, MagicMock]:
    (tmp_path / "figs" / "1").mkdir(parents=True)
    (tmp_path / "figs" / "1" / "figure.png").write_bytes(_PNG)
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {"observability_enabled": True}
    server_client.post = AsyncMock(return_value=_model_reply("resp-1"))
    return LabbenchVLMAgent(
        config=_agent_config(media_base_dir=str(tmp_path)), server_client=server_client
    ), server_client


def test_an_agent_session_sends_the_tasks_images_to_the_model_and_keeps_them_out_of_its_observations(
    session_agent,
) -> None:
    agent, server_client = session_agent
    client = TestClient(agent.setup_webserver())
    question = {"input": [{"role": "user", "content": [{"type": "input_text", "text": "What is shown?"}]}]}

    assert client.post("/v1/agent_sessions", json=_seed("figs/1").model_dump(mode="json")).status_code == 200
    assert client.post("/ng-rollout/rollout/v1/responses", json=question).status_code == 200
    close = client.post(
        "/v1/agent_sessions/close",
        json=AgentCloseSessionRequest(
            agent_session_id="labbench2-session", episode_id=EpisodeId(rollout_id="rollout", attempt=0)
        ).model_dump(mode="json"),
    )

    [model_call] = server_client.post.await_args_list
    content = model_call.kwargs["json"].model_dump(mode="json")["input"][0]["content"]
    assert [block["type"] for block in content] == ["input_image", "input_text"]
    assert content[0]["image_url"].startswith("data:image/png;base64,")
    assert content[1]["text"] == "What is shown?"
    observations = close.json()["agent_observations"]
    assert "input_image" not in json.dumps(observations)
    assert {"code": "multimodal_history_redacted"} in [
        {key: value for key, value in gap.items() if value is not None} for gap in observations["gaps"]
    ]


@pytest.mark.parametrize(
    ("media_path", "detail"),
    [
        (None, "agent_context does not satisfy LabbenchVLMAgentContext: media: Field required"),
        ("figs/missing", "agent_context artifact 'figs/missing' is not a directory under"),
    ],
)
def test_an_agent_session_without_its_media_is_refused(session_agent, media_path, detail) -> None:
    agent, server_client = session_agent

    seeded = TestClient(agent.setup_webserver()).post(
        "/v1/agent_sessions", json=_seed(media_path).model_dump(mode="json")
    )

    # 422 is terminal for an environment server: the episode fails once instead of running without images.
    assert seeded.status_code == 422
    assert seeded.json()["detail"].startswith(detail)
    server_client.post.assert_not_awaited()


def test_an_agent_session_without_the_tasks_tag_is_refused(session_agent) -> None:
    """The tag picks text or images for protocolqa2, so without it the model could get another prompt than /run."""
    agent, server_client = session_agent
    seed = _seed("figs/1").model_copy(update={"agent_context": AgentContext(artifacts={"media": {"path": "figs/1"}})})

    seeded = TestClient(agent.setup_webserver()).post("/v1/agent_sessions", json=seed.model_dump(mode="json"))

    assert seeded.status_code == 422
    assert seeded.json()["detail"] == "agent_context does not satisfy LabbenchVLMAgentContext: tag: Field required"
    server_client.post.assert_not_awaited()


_EXAMPLE_JSONL = Path(__file__).parents[3] / "resources_servers/labbench2_vlm/data/example.jsonl"
# The committed example rows by tag, with their media under the agent's default media_base_dir.
_EXAMPLE_ROWS = {
    row["verifier_metadata"]["tag"]: row for row in map(json.loads, _EXAMPLE_JSONL.read_text().splitlines())
}


def _recording_agent(media_mode: str) -> tuple[LabbenchVLMAgent, list[dict]]:
    """An agent whose model calls are recorded; the Resources Server accepts every seed and verify."""
    model_requests = []

    async def post(**kwargs) -> MagicMock:
        body = kwargs["json"]
        body = body.model_dump(mode="json") if hasattr(body, "model_dump") else body
        if kwargs["url_path"].endswith("/v1/responses"):
            model_requests.append(body)
            return _model_reply(f"resp-{len(model_requests)}")
        reply = MagicMock(status=200, cookies={}, ok=True)
        reply.read = AsyncMock(return_value=json.dumps(body | {"reward": 1.0}))
        return reply

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server_client.post = AsyncMock(side_effect=post)
    return LabbenchVLMAgent(config=_agent_config(media_mode=media_mode), server_client=server_client), model_requests


@pytest.mark.parametrize(
    ("tag", "media_mode", "blocks"),
    [
        ("protocolqa2", "text", ["input_text", "input_text"]),
        ("protocolqa2", "image", ["input_image", "input_text"]),
        ("figqa2-img", "text", ["input_image", "input_text"]),
    ],
)
def test_an_agent_session_sends_the_model_the_same_request_as_run(tag: str, media_mode: str, blocks: list) -> None:
    row = _EXAMPLE_ROWS[tag]
    run_agent, run_requests = _recording_agent(media_mode)
    assert TestClient(run_agent.setup_webserver()).post("/run", json=row).status_code == 200

    session_agent, session_requests = _recording_agent(media_mode)
    client = TestClient(session_agent.setup_webserver())
    # The seed carries the context single_agent_turn takes from the row, as in a real episode.
    agent_context = SingleAgentTurnTaskInput.model_validate(row).agent_seed_context
    seed = _seed(None).model_copy(update={"agent_context": agent_context})
    assert client.post("/v1/agent_sessions", json=seed.model_dump(mode="json")).status_code == 200
    assert client.post("/ng-rollout/rollout/v1/responses", json=row["responses_create_params"]).status_code == 200

    assert session_requests == run_requests
    assert [block["type"] for block in session_requests[0]["input"][0]["content"]][-2:] == blocks


def test_effective_media_mode_text_only_for_protocolqa2() -> None:
    assert _effective_media_mode("image", {"tag": "protocolqa2"}) == "image"
    assert _effective_media_mode("text", {"tag": "protocolqa2"}) == "text"
    assert _effective_media_mode("text", {"tag": "figqa2-img"}) == "image"
    assert _effective_media_mode("text", {"tag": "tableqa2-pdf"}) == "image"
    assert _effective_media_mode("text", None) == "image"


async def test_run_preserves_unset_model_during_media_embedding(monkeypatch) -> None:
    body = SimpleAgentRunRequest.model_validate(
        {
            "responses_create_params": {
                "input": [{"role": "user", "content": "What changed in the protocol?"}],
            },
            "verifier_metadata": {
                "tag": "protocolqa2",
                "media_dir": "test_media/protocols/example",
            },
        }
    )

    def fake_embed_media_into_row(row_dict, *args, **kwargs):
        assert "model" not in row_dict["responses_create_params"]
        return row_dict

    monkeypatch.setattr(labbench_agent_app, "embed_media_into_row", fake_embed_media_into_row)
    super_run = AsyncMock(return_value=SimpleAgentVerifyResponse.model_construct())
    monkeypatch.setattr(SimpleAgent, "run", super_run)

    agent = LabbenchVLMAgent(
        config=_agent_config(strip_images_from_output=False),
        server_client=MagicMock(spec=ServerClient),
    )
    await agent.run(MagicMock(), body)

    forwarded_body = super_run.call_args.args[1]
    assert "model" not in forwarded_body.responses_create_params.model_fields_set


def test_strip_images_removes_media_from_response_and_trajectory() -> None:
    image = {"type": "input_image", "image_url": "data:image/png;base64,cGF5bG9hZA==", "detail": "auto"}
    message = {"role": "user", "content": [{"type": "input_text", "text": "question"}, image]}
    result = SimpleAgentVerifyResponse.model_validate(
        {
            "responses_create_params": {"input": [message]},
            "response": {
                "id": "response-1",
                "created_at": 1,
                "model": "model",
                "object": "response",
                "output": [],
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
            },
            "reward": 1.0,
            "ng_trajectory": {
                "schema_version": "1.0",
                "task_id": "task",
                "rollout_id": "rollout",
                "invocations": [{"kind": "agent_invocation", "invocation_id": "root", "conversation": [message]}],
            },
        }
    )

    stripped = _strip_image_blocks(result)
    serialized = json.dumps(stripped.model_dump(mode="json"))

    assert "input_image" not in serialized
    assert "cGF5bG9hZA==" not in serialized
    assert "question" in serialized
    assert [gap["code"] for gap in stripped.model_dump(mode="json")["ng_trajectory"]["gaps"]] == [
        "multimodal_history_redacted"
    ]
    assert len(_strip_image_blocks(stripped).model_dump(mode="json")["ng_trajectory"]["gaps"]) == 1
