# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import json
from pathlib import PurePosixPath
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from relay.core import EndpointResult, ProducedFile, RelayEvent

from nemo_gym.client import ServerClient
from nemo_gym.config import ResourcesServerRef
from responses_api_agents.relay_agent.app import (
    RelayAgent,
    RelayAgentConfig,
    RelayAgentRunRequest,
    lifecycle_logger,
)


INSTRUCTION = "Write one JSON object per line to `results_smoke.jsonl`."


def make_agent(tmp_path=None, **config: object) -> RelayAgent:
    return RelayAgent(
        config=RelayAgentConfig(
            host="127.0.0.1",
            port=0,
            entrypoint="app.py",
            name="relay_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="wandr"),
            model="openai/openai/gpt-5.5",
            api_key="test-key",
            full_restart_initial_delay_s=0.0,
            trace_dir=str(tmp_path) if tmp_path else None,
            **config,
        ),
        server_client=MagicMock(spec=ServerClient),
    )


def mock_response(payload: dict) -> MagicMock:
    response = MagicMock()
    response.ok = True
    response.cookies = {}
    response.read = AsyncMock(return_value=json.dumps(payload).encode())
    return response


def resources_server(reward: float):
    """Seeds, then verifies by echoing the request with a reward, as the resources server does."""

    async def post(*, server_name: str, url_path: str, json: dict, cookies: dict) -> MagicMock:
        return mock_response({} if url_path == "/seed_session" else json | {"reward": reward})

    return post


def run_request() -> RelayAgentRunRequest:
    return RelayAgentRunRequest.model_validate(
        {
            "responses_create_params": {
                "input": [
                    {"role": "system", "content": "simple_agent tool instructions, not for this agent"},
                    {"role": "user", "content": INSTRUCTION},
                ]
            },
            "task_name": "smoke",
            "required_output_files": ["results_smoke.jsonl"],
        }
    )


class FakeEndpoint:
    """Stands in for the vendor endpoint; Relay itself (prompt, restarts, file checks) runs unchanged."""

    def __init__(self, delivery_method, results: list) -> None:
        self.delivery_method = delivery_method
        self.results = results
        self.prompts: list[str] = []

    async def run(self, prompt: str, observer=None) -> EndpointResult:
        self.prompts.append(prompt)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def install_endpoint(agent: RelayAgent, monkeypatch: pytest.MonkeyPatch, results: list) -> FakeEndpoint:
    fake = FakeEndpoint(agent.make_endpoint().delivery_method, results)
    object.__setattr__(agent, "make_endpoint", lambda: fake)
    return fake


def verify_payloads(agent: RelayAgent) -> list[dict]:
    return [c.kwargs["json"] for c in agent.server_client.post.call_args_list if c.kwargs["url_path"] == "/verify"]


def test_endpoint_sends_the_upstream_openai_request() -> None:
    """The request matches upstream configs/wandr.yaml: hosted search and code interpreter, background, effort high."""
    payload = make_agent().make_endpoint()._create_payload("prompt")
    assert payload["model"] == "openai/openai/gpt-5.5"
    assert payload["background"] is True
    assert payload["reasoning"]["effort"] == "high"
    assert [tool["type"] for tool in payload["tools"]] == ["web_search_preview", "code_interpreter"]


async def test_run_verifies_the_delivered_files(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent(tmp_path)
    content = b'{"item": {"topic": "GC"}}\n'
    fake = install_endpoint(
        agent,
        monkeypatch,
        [
            EndpointResult(
                text="done",
                files=(ProducedFile(path=PurePosixPath("results_smoke.jsonl"), content=content),),
                usage={"input_tokens": 10},
                response_id="resp_1",
            )
        ],
    )
    agent.server_client.post.side_effect = resources_server(0.25)
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request=request, body=run_request())

    assert result.reward == 0.25
    (prompt,) = fake.prompts
    assert prompt.rstrip().endswith(INSTRUCTION)
    assert "simple_agent tool instructions" not in prompt
    assert "- `results_smoke.jsonl`" in prompt
    (verify,) = verify_payloads(agent)
    assert verify["output_files"] == {"results_smoke.jsonl": content.decode()}
    assert verify["task_name"] == "smoke"
    assert verify["response"]["output"][0]["content"][0]["text"] == "done"
    assert result.model_extra["relay_error"] is None
    (trace,) = tmp_path.iterdir()
    assert json.loads(trace.read_text())["response_id"] == "resp_1"


def test_lifecycle_log_keeps_run_status_and_drops_tool_traffic(capsys: pytest.CaptureFixture) -> None:
    observer = lifecycle_logger("smoke")
    observer(RelayEvent(name="endpoint.submitted", message="Submitted.", metadata={"response_id": "resp_1"}))
    observer(RelayEvent(name="endpoint.tool_call", semantic_type="tool_call", metadata={"query": "q"}))
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 1
    assert "smoke" in lines[0] and "endpoint.submitted" in lines[0] and "resp_1" in lines[0]


async def test_run_restarts_then_scores_a_failed_run_without_files(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent(max_full_restarts=1)
    fake = install_endpoint(agent, monkeypatch, [RuntimeError("sandbox died"), EndpointResult(text="no files")])
    agent.server_client.post.side_effect = resources_server(0.0)
    request = MagicMock(spec=Request)
    request.cookies = {}

    result = await agent.run(request=request, body=run_request())

    assert len(fake.prompts) == 2
    (verify,) = verify_payloads(agent)
    assert verify["output_files"] == {}
    assert "did not produce any files" in result.model_extra["relay_error"]
