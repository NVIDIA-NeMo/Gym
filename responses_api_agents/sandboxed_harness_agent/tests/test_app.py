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
import json
from shlex import quote
from types import SimpleNamespace
from typing import Any, ClassVar, Optional
from unittest.mock import AsyncMock, MagicMock

from pytest import mark

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.sandboxed_harness_agent.app import (
    HarnessTranscript,
    SandboxedHarnessAgent,
    SandboxedHarnessAgentConfig,
    SandboxedHarnessAgentRunRequest,
)


class EchoAgent(SandboxedHarnessAgent):
    """A harness that is not OpenCode: its command only echoes the prompt."""

    harness_name: ClassVar[str] = "Echo"
    harness_id: ClassVar[str] = "echo"
    finished_marker: ClassVar[str] = "Echo run finished"

    async def _harness_command(self, request: Any, query: str, collect_observations: bool) -> tuple[str, Any]:
        return f'echo "Shell: $SHELL" && echo {quote(query)} && echo "{self.finished_marker}"', None

    async def _harness_collect(
        self,
        request: Any,
        sandbox: Any,
        state: Any,
        collect_observations: bool,
        observation_invocation_id: Optional[str],
    ) -> HarnessTranscript:
        return HarnessTranscript(output=[], usage=None, results_fpath=None, observations=None)


class _Response:
    ok = True

    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload
        self.cookies: dict[str, str] = {}

    async def json(self) -> dict[str, Any]:
        return self.payload

    async def read(self) -> bytes:
        return json.dumps(self.payload).encode()


class _RunRequest:
    def __init__(self) -> None:
        self._cookies: dict[str, str] = {}
        self.session = {SESSION_ID_KEY: "session-1"}
        self.state = SimpleNamespace()

    @property
    def cookies(self) -> dict[str, str]:
        return self._cookies


async def _run(agent_class: type[SandboxedHarnessAgent], exec_outcome: Any) -> tuple[dict[str, Any], list[str], Any]:
    """Run one rollout whose harness exec returns or raises `exec_outcome`."""
    posts: list[str] = []

    async def post(server_name: str, url_path: str, json: Any = None, cookies: Any = None) -> _Response:
        posts.append(url_path)
        if url_path == "/seed_session":
            return _Response({"sandbox_handle": "task-sandbox"})
        return _Response(json | {"reward": 0.0})

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server_client.post = AsyncMock(side_effect=post)
    agent = agent_class(
        config=SandboxedHarnessAgentConfig(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name="echo_agent",
            resources_server=ResourcesServerRef(type="resources_servers", name="task_server"),
            model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
            sandbox_provider="sandbox",
            sandbox_config={},
            sandbox_timeout=60,
        ),
        server_client=server_client,
    )
    sandbox = MagicMock()
    sandbox.exec = AsyncMock(side_effect=[exec_outcome])
    sandbox.stop = AsyncMock()
    agent._start_sandbox = AsyncMock(return_value=sandbox)
    body = SandboxedHarnessAgentRunRequest.model_validate(
        {"responses_create_params": {"input": [{"role": "user", "content": "hello"}]}}
    )

    result = await agent.run(_RunRequest(), body)

    sandbox.stop.assert_awaited_once()
    return result.model_dump(mode="json"), posts, sandbox.exec.await_args


async def test_finished_run_reports_the_harness_fields() -> None:
    stdout = "Shell: /bin/bash\nhello\nEcho run finished"
    result, posts, exec_call = await _run(EchoAgent, SimpleNamespace(stdout=stdout, stderr="", return_code=0))

    assert exec_call.kwargs == {
        "command": 'echo "Shell: $SHELL" && echo hello && echo "Echo run finished"',
        "timeout_s": 60,
    }
    assert posts == ["/seed_session", "/verify"]
    assert result["harness_finished"] is True
    assert result["echo_finished"] is True
    assert result["echo_run_stdout"] == stdout
    assert result["echo_export_found"] is False
    assert result["echo_results_fpath"] == ""
    assert [item["role"] for item in result["responses_create_params"]["input"]] == ["user"]


@mark.parametrize(
    "exec_outcome",
    [
        SimpleNamespace(stdout="Shell: /bin/bash\nhello", stderr="boom", return_code=1),
        RuntimeError("sandbox evicted"),
        TimeoutError("command timed out"),
    ],
    ids=("missing-marker", "exec-exception", "exec-timeout"),
)
async def test_unfinished_run_is_still_verified(exec_outcome: Any) -> None:
    result, posts, _ = await _run(EchoAgent, exec_outcome)

    assert posts == ["/seed_session", "/verify"]
    assert result["harness_finished"] is False
    assert result["echo_finished"] is False


async def test_marker_counts_only_after_the_last_shell_line() -> None:
    stdout = "Shell: /bin/bash\nEcho run finished\nShell: /bin/bash\nhello"
    result, _, _ = await _run(EchoAgent, SimpleNamespace(stdout=stdout, stderr="", return_code=0))

    assert result["harness_finished"] is False


async def test_system_prompt_is_prepended_to_the_result_input() -> None:
    class PromptedEchoAgent(EchoAgent):
        system_prompt: ClassVar[Optional[str]] = "You are echo."

    stdout = "Shell: /bin/bash\nhello\nEcho run finished"
    result, _, _ = await _run(PromptedEchoAgent, SimpleNamespace(stdout=stdout, stderr="", return_code=0))

    first_input = result["responses_create_params"]["input"][0]
    assert (first_input["role"], first_input["content"]) == ("system", "You are echo.")
