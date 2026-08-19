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
from unittest.mock import AsyncMock, MagicMock

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import ServerClient
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
