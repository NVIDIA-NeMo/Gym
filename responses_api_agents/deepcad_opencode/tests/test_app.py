# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import asyncio
import base64
import io
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import Request
from PIL import Image

from nemo_gym.server_runtime.base_server import SESSION_ID_KEY
from resources_servers.deepcad_opencode.app import DeepCADOpenCodeResourcesServer, DeepCADOpenCodeVerifyRequest
from responses_api_agents.deepcad_opencode import cad_tools
from responses_api_agents.deepcad_opencode.app import (
    DeepCADOpenCodeAgent,
    DeepCADOpenCodeConfig,
    DeepCADOpenCodeRunRequest,
)
from responses_api_agents.opencode_sandboxed_agent.app import OpenCodeSandboxedAgent


def image_url():
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), "blue").save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()


@pytest.fixture
def agent(tmp_path):
    config = DeepCADOpenCodeConfig.model_construct(
        artifacts_dir=str(tmp_path / "artifacts"), execution_failure_reward_zero=True
    )
    server = DeepCADOpenCodeAgent.model_construct(config=config, server_client=SimpleNamespace())
    server._sem = asyncio.Semaphore(1)
    server._sandbox_id_to_sandbox = {}
    server._sandbox_id_to_run_result = {}
    return server


@pytest.fixture
def body():
    return DeepCADOpenCodeRunRequest(
        verifier_metadata={"task_id": "part", "reference_step": "secret.step", "view_mode": "axis"},
        responses_create_params={
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "Reconstruct this."},
                        {"type": "input_image", "image_url": image_url(), "detail": "high"},
                    ],
                }
            ]
        },
    )


@pytest.mark.asyncio
async def test_workspace_only_gets_images_and_helpers(agent, body, tmp_path):
    sandbox = SimpleNamespace(upload=AsyncMock())
    prompt = await agent._prepare_workspace(sandbox, body, tmp_path)
    uploads = [call.args for call in sandbox.upload.await_args_list]
    assert "/workspace/target-0.png" in [r for _, r in uploads]
    assert all("secret.step" not in str(local) for local, _ in uploads)
    assert "secret.step" not in prompt and "model.py" in prompt and "cad_view" in prompt
    assert Image.open(tmp_path / "target-0.png").size == (256, 256)
    assert "import cadquery as cq" in prompt


@pytest.mark.asyncio
async def test_config_disables_compaction_and_delegation(agent, monkeypatch):
    monkeypatch.setattr(
        OpenCodeSandboxedAgent,
        "_create_opencode_config",
        AsyncMock(return_value={"compaction": {"auto": True}, "permission": {"task": "allow"}}),
    )
    config = await agent._create_opencode_config(None)
    assert config["compaction"] == {"auto": False, "prune": False}
    assert config["permission"]["task"] == "deny"
    assert config["agent"]["explore"]["disable"] and config["agent"]["general"]["disable"]
    assert config["mcp"]["cad"]["command"][-1].endswith("cad_tools.py")
    assert config["provider"]["nemo_gym"]["models"]["dummy_model"]["modalities"]["input"] == ["text", "image"]


def test_tool_image_is_preserved_in_training_response(agent):
    url = image_url()
    export = {
        "messages": [
            {
                "info": {"role": "assistant"},
                "parts": [
                    {
                        "type": "tool",
                        "callID": "call1",
                        "tool": "cad_view",
                        "state": {
                            "input": {"path": "preview.png"},
                            "output": "Preview",
                            "attachments": [{"mime": "image/png", "url": url}],
                        },
                    }
                ],
            }
        ]
    }
    output = agent._opencode_export_to_output_items(export)
    assert output[1].type == "function_call_output"
    assert output[1].output[1]["image_url"] == url
    assert output[1].output[0]["text"] == "Preview"


def test_verifier_uses_final_file_even_if_answer_contains_other_code():
    req = DeepCADOpenCodeVerifyRequest.model_construct(
        final_code='raise ValueError("latest invalid code")', response=None
    )
    assert DeepCADOpenCodeResourcesServer.code_from_response(None, req) == req.final_code


@pytest.mark.asyncio
async def test_missing_final_file_is_zero_source(agent, tmp_path):
    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(error_type=None, return_code=0, stdout="False", stderr="")),
        download=AsyncMock(),
    )
    assert await agent._collect_code(sandbox, tmp_path) == ("", "missing_or_invalid_model_file")
    sandbox.download.assert_not_awaited()


@pytest.mark.asyncio
async def test_final_file_is_collected_not_answer(agent, tmp_path):
    async def download(remote, local):
        assert remote == "/workspace/model.py"
        local.write_text("import cadquery as cq\nresult = cq.Workplane().box(1,2,3)")

    sandbox = SimpleNamespace(
        exec=AsyncMock(return_value=SimpleNamespace(error_type=None, return_code=0, stdout="True", stderr="")),
        download=download,
    )
    code, status = await agent._collect_code(sandbox, tmp_path)
    assert status == "collected" and "box(1,2,3)" in code


def test_self_created_image_feedback_and_path_boundary(tmp_path, monkeypatch):
    monkeypatch.setattr(cad_tools, "ROOT", tmp_path)
    Image.new("RGB", (64, 32), "green").save(tmp_path / "own.png")
    result = cad_tools.call_tool("cad_view", {"path": "own.png"})
    assert result["content"][1]["type"] == "image"
    decoded = Image.open(io.BytesIO(base64.b64decode(result["content"][1]["data"])))
    assert decoded.size == (64, 32)
    assert cad_tools.call_tool("cad_view", {"path": "../outside.png"})["isError"]


def test_failed_render_does_not_return_stale_images(tmp_path, monkeypatch):
    monkeypatch.setattr(cad_tools, "ROOT", tmp_path)
    Image.new("RGB", (32, 32), "red").save(tmp_path / "old.png")

    class Process:
        returncode = 1

        def communicate(self, timeout):
            return b"", b"Traceback: NameError: cq is not defined"

    monkeypatch.setattr(cad_tools.subprocess, "Popen", lambda *a, **k: Process())
    result = cad_tools.call_tool("cad_render", {})
    assert result["isError"] and "NameError" in result["content"][0]["text"]
    assert all(c["type"] == "text" for c in result["content"])


@pytest.mark.asyncio
@pytest.mark.parametrize("failed", [False, True])
async def test_episode_grades_final_file_and_preserves_limit_status(agent, body, monkeypatch, failed):
    import responses_api_agents.deepcad_opencode.app as app

    sandbox = SimpleNamespace(stop=AsyncMock())
    request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "episode"}})
    response = SimpleNamespace(
        model_dump=lambda **kwargs: {"id": "answer", "output": [], "status": "incomplete" if failed else "completed"}
    )
    agent.config.resources_server = SimpleNamespace(name="verifier")
    monkeypatch.setattr(DeepCADOpenCodeAgent, "rollout_id_from_run", lambda *args: None)
    monkeypatch.setattr(DeepCADOpenCodeAgent, "_start_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(DeepCADOpenCodeAgent, "_prepare_workspace", AsyncMock(return_value="Build"))
    monkeypatch.setattr(DeepCADOpenCodeAgent, "_collect_code", AsyncMock(return_value=("FINAL FILE", "collected")))

    async def respond(self, request, params):
        # Upstream OpenCode receives both the CAD instructions and actual target images.
        parts = params.model_dump()["input"][0]["content"]
        assert any(part["type"] == "input_image" and part["image_url"] == image_url() for part in parts)
        assert any(part["type"] == "input_text" and part["text"] == "Build" for part in parts)
        agent._sandbox_id_to_run_result["episode"] = {"opencode_failed": failed}
        return response

    monkeypatch.setattr(DeepCADOpenCodeAgent, "responses", respond)
    post = AsyncMock(return_value=object())
    agent.server_client = SimpleNamespace(post=post)
    monkeypatch.setattr(app, "raise_for_status", AsyncMock())
    monkeypatch.setattr(app, "get_response_json", AsyncMock(return_value={"reward": 0, "mask_sample": False}))
    result = await agent.run(request, body)
    assert post.call_args.kwargs["json"]["final_code"] == ("" if failed else "FINAL FILE")
    assert result["response"]["status"] == ("incomplete" if failed else "completed")
    assert result["mask_sample"] is False
    sandbox.stop.assert_awaited_once()
    assert not agent._sandbox_id_to_sandbox and not agent._sandbox_id_to_run_result


@pytest.mark.asyncio
async def test_episode_cleanup_on_model_error(agent, body, monkeypatch):
    sandbox = SimpleNamespace(stop=AsyncMock())
    request = Request({"type": "http", "headers": [], "session": {SESSION_ID_KEY: "episode"}})
    monkeypatch.setattr(DeepCADOpenCodeAgent, "rollout_id_from_run", lambda *args: None)
    monkeypatch.setattr(DeepCADOpenCodeAgent, "_start_sandbox", AsyncMock(return_value=sandbox))
    monkeypatch.setattr(DeepCADOpenCodeAgent, "_prepare_workspace", AsyncMock(return_value="Build"))
    monkeypatch.setattr(DeepCADOpenCodeAgent, "responses", AsyncMock(side_effect=RuntimeError("model offline")))
    with pytest.raises(RuntimeError, match="model offline"):
        await agent.run(request, body)
    sandbox.stop.assert_awaited_once()
    assert not agent._sandbox_id_to_sandbox
