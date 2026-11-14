# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import httpx
import pytest


pytest.importorskip("harbor.agents.base")
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories import Trajectory

from benchmarks.vials.harbor.agent import VialsVisionAgent


class FilesEnvironment:
    def __init__(self, files):
        self.files = files
        self.downloads = []

    async def download_file(self, source, destination):
        self.downloads.append(source)
        Path(destination).write_bytes(self.files[source])

    async def upload_file(self, source, destination):
        self.files[destination] = Path(source).read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [32768, 131072, 262144])
async def test_single_vision_request_and_atif(tmp_path, monkeypatch, budget):
    requests = []

    def backend(request):
        body = json.loads(request.content)
        requests.append(body)
        assert "tools" not in body
        assert body["tool_choice"] == "none"
        assert body["max_completion_tokens"] == budget
        assert body["chat_template_kwargs"] == {"enable_thinking": True}
        assert [m["role"] for m in body["messages"]] == ["system", "user"]
        assert "REFERENCE_SENTINEL" not in request.content.decode()
        images = body["messages"][1]["content"][1:]
        assert [p["image_url"]["url"] for p in images] == [
            "data:image/png;base64,Zmlyc3Q=",
            "data:image/png;base64,c2Vjb25k",
        ]
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": "ANSWER: A", "reasoning_content": "reason"}, "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    original = httpx.AsyncClient
    monkeypatch.setattr(
        httpx, "AsyncClient", lambda **kwargs: original(transport=httpx.MockTransport(backend), **kwargs)
    )
    public = {"system": "scientist", "user": "question", "images": ["a.png", "b.png"]}
    environment = FilesEnvironment(
        {
            "/task/vials_input.json": json.dumps(public).encode(),
            "/task/a.png": b"first",
            "/task/b.png": b"second",
            "/tests/reference": b"REFERENCE_SENTINEL",
        }
    )
    agent = VialsVisionAgent(
        logs_dir=tmp_path,
        model_name="vision-model",
        max_output_tokens=budget,
        chat_template_kwargs={"enable_thinking": True},
        extra_env={"VIALS_POLICY_BASE_URL": "http://policy/v1"},
    )
    context = AgentContext()
    await agent.setup(environment)
    await agent.run("instruction", environment, context)
    assert len(requests) == 1
    assert environment.files["/task/answer.txt"] == b"ANSWER: A"
    assert environment.downloads == ["/task/vials_input.json", "/task/a.png", "/task/b.png"]
    trajectory = Trajectory.model_validate_json((tmp_path / "trajectory.json").read_text())
    assert [s.source for s in trajectory.steps] == ["system", "user", "agent"]
    assert trajectory.steps[-1].llm_call_count == 1
    assert not trajectory.steps[-1].tool_calls
    assert trajectory.steps[-1].reasoning_content == "reason"
    assert context.n_output_tokens == 5 and context.n_input_tokens == 10


@pytest.mark.asyncio
async def test_tool_response_fails_without_publishing_answer(tmp_path, monkeypatch):
    original = httpx.AsyncClient
    transport = httpx.MockTransport(
        lambda request: httpx.Response(200, json={"choices": [{"message": {"tool_calls": [{"id": "forbidden"}]}}]})
    )
    monkeypatch.setattr(httpx, "AsyncClient", lambda **kwargs: original(transport=transport, **kwargs))
    environment = FilesEnvironment(
        {"/task/vials_input.json": json.dumps({"user": "q", "images": ["a.png"]}).encode(), "/task/a.png": b"image"}
    )
    agent = VialsVisionAgent(
        logs_dir=tmp_path, model_name="vision-model", extra_env={"VIALS_POLICY_BASE_URL": "http://policy/v1"}
    )
    with pytest.raises(ValueError, match="tool call"):
        await agent.run("instruction", environment, AgentContext())
    assert "/task/answer.txt" not in environment.files
