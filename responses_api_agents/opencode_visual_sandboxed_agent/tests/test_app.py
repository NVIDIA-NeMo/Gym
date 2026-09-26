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
import base64
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymEasyInputMessage,
    NeMoGymFunctionCallOutput,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
)
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_visual_sandboxed_agent import app as app_module
from responses_api_agents.opencode_visual_sandboxed_agent.app import (
    TOOL_MEDIA_PREAMBLE,
    OpenCodeVisualSandboxedAgent,
    OpenCodeVisualSandboxedAgentConfig,
    decode_image_data_url,
    split_prompt_images,
)


PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_URL = "data:image/png;base64," + base64.b64encode(PNG_BYTES).decode()


def make_agent(**overrides: Any) -> OpenCodeVisualSandboxedAgent:
    config = OpenCodeVisualSandboxedAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="",
        resources_server=ResourcesServerRef(type="resources_servers", name=""),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        opencode_version="",
        sandbox_provider="",
        sandbox_config={},
        sandbox_timeout=0,
        opencode_max_context_window=1000,
        **overrides,
    )
    return OpenCodeVisualSandboxedAgent(config=config, server_client=MagicMock(spec=ServerClient))


class TestPromptImages:
    def test_decode_image_data_url(self) -> None:
        mime, data = decode_image_data_url(PNG_URL)
        assert mime == "image/png"
        assert data == PNG_BYTES

    @pytest.mark.parametrize(
        "url",
        [
            "https://example.com/a.png",
            "data:text/plain;base64,aGVsbG8=",
            "data:image/png,notbase64",
            "data:image/png;base64,***",
        ],
    )
    def test_decode_image_data_url_rejects(self, url: str) -> None:
        with pytest.raises(ValueError):
            decode_image_data_url(url)

    def test_split_prompt_images_keeps_order_and_joins_text(self) -> None:
        items = [
            {"role": "system", "content": "sys"},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Recreate this."},
                    {"type": "input_image", "image_url": PNG_URL, "detail": "high"},
                    {"type": "input_text", "text": "Match colors."},
                    {"type": "input_image", "image_url": PNG_URL + "AA", "detail": "high"},
                ],
            },
        ]
        rewritten, images = split_prompt_images(items)
        assert rewritten[0] == items[0]
        assert rewritten[1] == NeMoGymEasyInputMessage(role="user", content="Recreate this.\n\nMatch colors.")
        assert images == [PNG_URL, PNG_URL + "AA"]

    def test_split_prompt_images_passthrough_for_text(self) -> None:
        items = [{"role": "user", "content": "just text"}]
        assert split_prompt_images(items) == (items, [])

    def test_split_prompt_images_rejects_file_id(self) -> None:
        with pytest.raises(ValueError, match="image_url"):
            split_prompt_images([{"role": "user", "content": [{"type": "input_image", "file_id": "f1"}]}])


class TestOpenCodeConfig:
    async def test_declares_image_modality(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agent()
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.get_server_url", lambda _name: "http://model:1"
        )
        request = MagicMock()
        request.json = AsyncMock(return_value={})
        config = await agent._create_opencode_config(request)
        model = config["provider"]["nemo_gym"]["models"]["dummy_model"]
        assert model["modalities"] == {"input": ["text", "image"], "output": ["text"]}
        assert model["attachment"] is True

    async def test_image_modality_can_be_disabled(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agent(enable_image_input=False)
        monkeypatch.setattr(
            "responses_api_agents.opencode_sandboxed_agent.app.get_server_url", lambda _name: "http://model:1"
        )
        request = MagicMock()
        request.json = AsyncMock(return_value={})
        config = await agent._create_opencode_config(request)
        assert "modalities" not in config["provider"]["nemo_gym"]["models"]["dummy_model"]


class TestResponses:
    async def test_uploads_prompt_images_and_attaches_them(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agent()
        sandbox = MagicMock()
        sandbox.exec = AsyncMock(return_value=SimpleNamespace(return_code=0))
        uploaded: Dict[str, bytes] = {}

        async def upload(local: Path, remote: str) -> None:
            uploaded[remote] = Path(local).read_bytes()

        sandbox.upload = upload
        agent._sandbox_id_to_sandbox["sb"] = sandbox
        seen: Dict[str, Any] = {}

        async def fake_parent_responses(self: Any, request: Any, body: Any) -> str:
            seen["extra_args"] = self._opencode_run_extra_args(request)
            seen["input"] = body.input
            return "response"

        monkeypatch.setattr(app_module.OpenCodeSandboxedAgent, "responses", fake_parent_responses)
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Recreate it."},
                        {"type": "input_image", "image_url": PNG_URL, "detail": "high"},
                    ],
                }
            ]
        )
        request = MagicMock(cookies={"sandbox_id": "sb"}, session={SESSION_ID_KEY: "s"})
        assert await agent.responses(request, body) == "response"

        remote = "/tmp/nemo_gym_prompt_images/prompt_image_0.png"
        assert uploaded == {remote: PNG_BYTES}
        assert seen["extra_args"] == f"--file {remote}"
        assert seen["input"] == [NeMoGymEasyInputMessage(role="user", content="Recreate it.")]
        # The per-request attachment list is cleared afterwards.
        assert agent._sandbox_id_to_prompt_files == {}

    async def test_text_only_prompt_has_no_file_args(self, monkeypatch: pytest.MonkeyPatch) -> None:
        agent = make_agent()
        agent._sandbox_id_to_sandbox["sb"] = MagicMock()
        seen: Dict[str, Any] = {}

        async def fake_parent_responses(self: Any, request: Any, body: Any) -> str:
            seen["extra_args"] = self._opencode_run_extra_args(request)
            return "response"

        monkeypatch.setattr(app_module.OpenCodeSandboxedAgent, "responses", fake_parent_responses)
        body = NeMoGymResponseCreateParamsNonStreaming(input=[{"role": "user", "content": "hello"}])
        await agent.responses(MagicMock(cookies={"sandbox_id": "sb"}), body)
        assert seen["extra_args"] == ""

    async def test_oversized_prompt_image_is_rejected(self) -> None:
        agent = make_agent(max_prompt_image_bytes=10)
        sandbox = MagicMock()
        sandbox.exec = AsyncMock(return_value=SimpleNamespace(return_code=0))
        sandbox.upload = AsyncMock()
        agent._sandbox_id_to_sandbox["sb"] = sandbox
        big = "data:image/png;base64," + base64.b64encode(b"x" * 11).decode()
        body = NeMoGymResponseCreateParamsNonStreaming(
            input=[{"role": "user", "content": [{"type": "input_image", "image_url": big, "detail": "auto"}]}]
        )
        with pytest.raises(ValueError, match="max_prompt_image_bytes"):
            await agent.responses(MagicMock(cookies={"sandbox_id": "sb"}), body)
        sandbox.upload.assert_not_awaited()


def _export(user_parts: List[Dict[str, Any]], tool_attachments: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "messages": [
            {"info": {"role": "user"}, "parts": user_parts},
            {
                "info": {"role": "assistant"},
                "parts": [
                    {"type": "step-start"},
                    {
                        "type": "tool",
                        "tool": "read",
                        "callID": "call_1",
                        "state": {
                            "status": "completed",
                            "input": {"filePath": "/workspace/previews/desktop.png"},
                            "output": "Image read successfully",
                            "attachments": tool_attachments,
                        },
                    },
                    {"type": "step-finish"},
                ],
            },
        ]
    }


class TestExportConversion:
    def test_tool_image_attachments_become_user_image_message(self) -> None:
        agent = make_agent(rollout_image_mode="inline")
        export = _export(
            [{"type": "text", "text": "task"}],
            [
                {"type": "file", "mime": "image/png", "url": PNG_URL},
                {"type": "file", "mime": "text/plain", "url": "data:text/plain;base64,aGk="},
            ],
        )
        items = agent._opencode_export_to_output_items(export)
        assert items[0] == NeMoGymEasyInputMessage(role="user", content=[{"type": "input_text", "text": "task"}])
        assert items[1] == NeMoGymResponseFunctionToolCall(
            arguments=json.dumps({"filePath": "/workspace/previews/desktop.png"}), call_id="call_1", name="read"
        )
        assert items[2] == NeMoGymFunctionCallOutput(call_id="call_1", output="Image read successfully")
        assert items[3] == NeMoGymEasyInputMessage(
            role="user",
            content=[
                {"type": "input_text", "text": TOOL_MEDIA_PREAMBLE},
                {"type": "input_image", "image_url": PNG_URL, "detail": "auto"},
            ],
        )
        assert len(items) == 4

    def test_prompt_file_parts_and_reference_mode(self) -> None:
        agent = make_agent(rollout_image_mode="reference")
        export = _export(
            [
                {"type": "text", "text": "Called the Read tool", "synthetic": True},
                {"type": "file", "mime": "image/png", "url": PNG_URL, "filename": "ref.png"},
                {"type": "text", "text": "task"},
            ],
            [],
        )
        items = agent._opencode_export_to_output_items(export)
        image_part = items[0].content[1]
        assert image_part["type"] == "input_image"
        assert image_part["image_url"].startswith("sha256:") and image_part["image_url"].endswith(";mime=image/png")
        assert PNG_URL not in json.dumps([i if isinstance(i, dict) else i.model_dump() for i in items])
        # No attachments on the tool result: no extra user message.
        assert [type(i).__name__ for i in items] == [
            "NeMoGymEasyInputMessage",
            "NeMoGymResponseFunctionToolCall",
            "NeMoGymFunctionCallOutput",
        ]
