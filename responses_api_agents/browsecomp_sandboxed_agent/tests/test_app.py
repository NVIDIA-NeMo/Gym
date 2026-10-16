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
"""The sandboxed agent runs bash_command in its sandbox and everything else as before.

The loop is the real BrowsecompAgent loop driven by a fake model; the sandbox is
the `local` provider, so pod_bash.py really executes against the synced pages.
"""

import json as jsonlib
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseFunctionToolCall,
    NeMoGymResponseInputTokensDetails,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
    NeMoGymResponseOutputTokensDetails,
    NeMoGymResponseUsage,
)
from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from responses_api_agents.browsecomp_agent.app import BrowsecompAgent
from responses_api_agents.browsecomp_sandboxed_agent.app import (
    SANDBOX_KEY_METADATA,
    BrowsecompSandboxedAgent,
    BrowsecompSandboxedAgentConfig,
    _RolloutSandbox,
)


USER = {"role": "user", "content": "Q?"}
PAGE_1 = ("pages/0001_browse_alpha.txt", "[URL]: https://a.example\n[Title]: A\n\nDavid Pemantle 1983\n")
PAGE_2 = ("pages/0002_browse_beta.txt", "[URL]: https://b.example\n[Title]: B\n\nThesis 1994\n")


def _fn_call(name: str, args: dict, call_id: str) -> NeMoGymResponseFunctionToolCall:
    return NeMoGymResponseFunctionToolCall(
        id=f"fc_{call_id}", call_id=call_id, name=name, arguments=jsonlib.dumps(args), type="function_call"
    )


def _answer() -> NeMoGymResponseOutputMessage:
    return NeMoGymResponseOutputMessage(
        id="msg_final",
        content=[NeMoGymResponseOutputText(annotations=[], text="Exact Answer: X", type="output_text")],
        role="assistant",
        status="completed",
        type="message",
    )


def _model_response(outputs: list, response_id: str) -> dict:
    return NeMoGymResponse(
        id=response_id,
        created_at=0.0,
        model="test_model",
        object="response",
        output=outputs,
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
        usage=NeMoGymResponseUsage(
            input_tokens=0,
            input_tokens_details=NeMoGymResponseInputTokensDetails(cached_tokens=0),
            output_tokens=0,
            output_tokens_details=NeMoGymResponseOutputTokensDetails(reasoning_tokens=0),
            total_tokens=0,
        ),
    ).model_dump()


class _FakeServerClient:
    """Serves model turns and the resources server's page workspace."""

    def __init__(self, model_turns: list, pages_by_call: list, enabled: bool = True):
        self._model_turns = [_model_response(t, f"resp_{i}") for i, t in enumerate(model_turns)]
        self._pages = []
        self._pages_by_call = list(pages_by_call)  # pages that exist before the Nth sync
        self._enabled = enabled
        self.model_bodies = []
        self.tool_paths = []
        self.sync_sinces = []

    async def post(self, server_name=None, url_path=None, json=None, cookies=None):
        http = MagicMock()
        http.ok = True
        http.status = 200
        http.cookies = {}
        if url_path == "/v1/responses":
            self.model_bodies.append(json)
            http.read = AsyncMock(return_value=jsonlib.dumps(self._model_turns.pop(0)).encode())
        elif url_path == "/workspace_files":
            self.sync_sinces.append(json["since"])
            if self._pages_by_call:
                self._pages = self._pages_by_call.pop(0)
            new = self._pages[json["since"] :]
            files = [{"path": p, "content": c} for p, c in new]
            if new:
                files.append({"path": "manifest.tsv", "content": "page_id\tkind\tsource\ttitle\tbytes\n"})
            payload = {"enabled": self._enabled, "files": files, "next": len(self._pages)}
            http.read = AsyncMock(return_value=jsonlib.dumps(payload).encode())
        else:
            self.tool_paths.append(url_path)
            http.content.read = AsyncMock(return_value=b'{"results_string": "host tool result"}')
        return http


def _make_agent(tmp_path, fake: _FakeServerClient) -> BrowsecompSandboxedAgent:
    config = BrowsecompSandboxedAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="",
        name="test_agent",
        resources_server=ResourcesServerRef(type="resources_servers", name="test_resources"),
        model_server=ModelServerRef(type="responses_api_models", name="test_model"),
        nudge_steps=False,
        sandbox_provider={"local": {"workspace_root": str(tmp_path / "local_provider")}},
        sandbox_config={},
        sandbox_root=str(tmp_path / "pod"),
        sandbox_python=sys.executable,
    )
    server_client = MagicMock(spec=ServerClient)
    server_client.post = fake.post
    return BrowsecompSandboxedAgent(config=config, server_client=server_client)


async def _responses(agent: BrowsecompSandboxedAgent, key: str) -> NeMoGymResponse:
    request = MagicMock()
    request.cookies = {}
    body = NeMoGymResponseCreateParamsNonStreaming(input=[USER], metadata={SANDBOX_KEY_METADATA: key})
    return await agent.responses(request, MagicMock(), body)


def _tool_outputs(result: NeMoGymResponse) -> list[str]:
    return [o.output for o in result.output if getattr(o, "type", None) == "function_call_output"]


@pytest.fixture
async def rollout(request):
    """Register one rollout's sandbox state and stop its sandbox afterwards."""
    states = []

    def _register(agent: BrowsecompSandboxedAgent, key: str = "k1") -> _RolloutSandbox:
        state = agent._rollouts[key] = _RolloutSandbox()
        states.append(state)
        return state

    yield _register
    for state in states:
        if state.sandbox is not None:
            await state.sandbox.stop()


async def test_bash_runs_in_sandbox_on_synced_pages(tmp_path, rollout) -> None:
    fake = _FakeServerClient(
        [[_fn_call("bash_command", {"keystrokes": "grep -h David pages/*.txt"}, "c1")], [_answer()]],
        pages_by_call=[[PAGE_1]],
    )
    agent = _make_agent(tmp_path, fake)
    rollout(agent)

    result = await _responses(agent, "k1")

    assert _tool_outputs(result) == ["David Pemantle 1983\n\n[exit_code=0]"]
    assert fake.tool_paths == []  # bash never reached the resources server
    assert SANDBOX_KEY_METADATA not in (fake.model_bodies[0].metadata or {})


async def test_one_sandbox_per_rollout_with_incremental_sync(tmp_path, rollout) -> None:
    fake = _FakeServerClient(
        [
            [_fn_call("bash_command", {"keystrokes": "ls pages"}, "c1")],
            [_fn_call("bash_command", {"keystrokes": "grep -h 1994 pages/*.txt"}, "c2")],
            [_answer()],
        ],
        pages_by_call=[[PAGE_1], [PAGE_1, PAGE_2]],
    )
    agent = _make_agent(tmp_path, fake)
    state = rollout(agent)

    result = await _responses(agent, "k1")

    assert _tool_outputs(result) == ["0001_browse_alpha.txt\n\n[exit_code=0]", "Thesis 1994\n\n[exit_code=0]"]
    assert fake.sync_sinces == [0, 1]
    assert state.synced == 2
    assert len(list((tmp_path / "local_provider").iterdir())) == 1


async def test_other_tools_go_to_the_resources_server(tmp_path, rollout) -> None:
    fake = _FakeServerClient([[_fn_call("search", {"queries": ["q"]}, "c1")], [_answer()]], pages_by_call=[])
    agent = _make_agent(tmp_path, fake)
    state = rollout(agent)

    result = await _responses(agent, "k1")

    assert _tool_outputs(result) == ["host tool result"]
    assert fake.tool_paths == ["/search"]
    assert state.sandbox is None  # no bash call, no sandbox


async def test_disabled_workspace_matches_host_message(tmp_path, rollout) -> None:
    fake = _FakeServerClient(
        [[_fn_call("bash_command", {"keystrokes": "ls"}, "c1")], [_answer()]], pages_by_call=[], enabled=False
    )
    agent = _make_agent(tmp_path, fake)
    state = rollout(agent)

    result = await _responses(agent, "k1")

    assert _tool_outputs(result) == ["[bash error: terminal workspace is not enabled for this server]"]
    assert state.sandbox is None


async def test_sandbox_timeout_matches_host_message(tmp_path, rollout) -> None:
    fake = _FakeServerClient(
        [[_fn_call("bash_command", {"keystrokes": "ls", "duration": 5}, "c1")], [_answer()]], pages_by_call=[]
    )
    agent = _make_agent(tmp_path, fake)
    state = rollout(agent)
    state.sandbox = MagicMock()
    state.sandbox.exec = AsyncMock(return_value=SandboxExecResult("", "", -1, error_type="timeout"))
    state.sandbox.stop = AsyncMock()

    result = await _responses(agent, "k1")

    assert _tool_outputs(result) == ["[command timed out after 5s]\n[exit_code=-1]"]


async def test_run_tags_the_rollout_and_stops_its_sandbox(tmp_path, monkeypatch) -> None:
    agent = _make_agent(tmp_path, _FakeServerClient([], pages_by_call=[]))
    seen = {}
    stub = MagicMock()
    stub.stop = AsyncMock()

    async def fake_base_run(self, request, body):
        key = body.responses_create_params.metadata[SANDBOX_KEY_METADATA]
        seen["key"] = key
        self._rollouts[key].sandbox = stub
        return "verify-response"

    monkeypatch.setattr(BrowsecompAgent, "run", fake_base_run)
    body = MagicMock()
    body.responses_create_params.metadata = None

    assert await agent.run(MagicMock(), body) == "verify-response"
    assert seen["key"]
    stub.stop.assert_awaited_once()
    assert agent._rollouts == {}
