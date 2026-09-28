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
"""Golden /run output of the OpenCode agent.

golden_run.json pins every sandbox exec call, the OpenCode config and the full /run result
for four ways a rollout can end. Re-record with NEMO_GYM_UPDATE_GOLDEN=1, only for an
intended change.
"""

import json
import os
import re
import shlex
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from pytest import MonkeyPatch, mark

from nemo_gym.config_types import ModelServerRef, ResourcesServerRef
from nemo_gym.sandbox import SandboxHandle
from nemo_gym.server_utils import SESSION_ID_KEY, ServerClient
from responses_api_agents.opencode_sandboxed_agent.app import (
    OpenCodeSandboxedAgent,
    OpenCodeSandboxedAgentConfig,
    OpenCodeSandboxedAgentRunRequest,
    OpenCodeSandboxedAgentVerifyResponse,
)


GOLDEN_PATH = Path(__file__).parent / "golden_run.json"
EXPORT = json.loads((Path(__file__).parent / "opencode_export_test_data.json").read_text())
RUN_BODY = {
    # A leading "--" must reach OpenCode as part of the prompt, not as an option.
    "responses_create_params": {"input": [{"role": "user", "content": "-- make the failing test pass"}]},
    "_ng_task_index": 7,
    "_ng_rollout_index": 2,
}
SCENARIOS = ("finished", "missing_marker", "exec_exception", "exec_timeout")


def _config() -> OpenCodeSandboxedAgentConfig:
    return OpenCodeSandboxedAgentConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="tb_opencode",
        resources_server=ResourcesServerRef(type="resources_servers", name="tb_resources"),
        model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        token_id_capture=True,
        opencode_version="1.17.11",
        opencode_max_context_window=196608,
        remote_opencode_install_script_path="/mnt/s3-data/data/harness/opencode/install.sh",
        remote_opencode_binary_path="/mnt/s3-data/data/harness/opencode/linux-x64/1.17.11/opencode",
        opencode_config={
            "permission": {"*": "allow", "bash": {"*": "allow", "*git fetch*": "deny"}},
            "tools": {"task": False, "webfetch": False},
            "compaction": {"auto": False, "prune": False},
        },
        sandbox_provider="sandbox",
        sandbox_config={},
        sandbox_timeout=3600,
    )


def _exec_result(stdout: str = "", stderr: str = "", return_code: int = 0) -> SimpleNamespace:
    return SimpleNamespace(stdout=stdout, stderr=stderr, return_code=return_code, error_type=None)


def _sandbox(scenario: str, downloads: list[str]) -> MagicMock:
    """A task sandbox whose OpenCode run ends the way `scenario` says."""

    async def fake_exec(command: str, **_: Any) -> SimpleNamespace:
        if "opencode run" in command:
            if scenario == "exec_exception":
                raise RuntimeError("sandbox evicted")
            if scenario == "exec_timeout":
                raise TimeoutError("command timed out")
            if scenario == "missing_marker":
                return _exec_result("Shell: /bin/bash\nInstalled OpenCode", "Error: provider failed", 1)
            return _exec_result("Shell: /bin/bash\nInstalled OpenCode\nOpenCode run finished")
        if scenario == "exec_exception":
            raise RuntimeError("sandbox evicted")
        if "session list" in command:
            return _exec_result("[]" if scenario == "missing_marker" else '[{"id": "ses_1"}]')
        if "opencode export" in command:
            return _exec_result()
        return _exec_result(stderr="python3: not found", return_code=127)  # the observation snapshot

    async def fake_download(remote_path: str, local_path: Path) -> None:
        downloads.append(remote_path)
        local_path.write_text(json.dumps(EXPORT))

    sandbox = MagicMock()
    sandbox._handle = SandboxHandle(sandbox_id="task-sandbox", provider_name="opensandbox", raw=None)
    sandbox.exec = AsyncMock(side_effect=fake_exec)
    sandbox.download = AsyncMock(side_effect=fake_download)
    sandbox.stop = AsyncMock()
    return sandbox


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

    async def json(self) -> dict[str, Any]:
        return RUN_BODY


async def _run(scenario: str, tmp_path: Path, monkeypatch: MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(
        "responses_api_agents.opencode_sandboxed_agent.app.get_server_url", lambda _name: "http://policy-model:8000"
    )
    monkeypatch.setattr("responses_api_agents.opencode_sandboxed_agent.app.__file__", str(tmp_path / "app.py"))
    monkeypatch.setattr("nemo_gym.responses_converter.uuid4", MagicMock(return_value=MagicMock(hex="0")))

    posts: list[str] = []

    async def post(server_name: str, url_path: str, json: Any = None, cookies: Any = None) -> _Response:
        posts.append(f"{server_name}{url_path}")
        if url_path == "/seed_session":
            return _Response({"sandbox_handle": "task-sandbox"})
        return _Response(json | {"reward": 0.0})

    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {"token_id_capture": {"enabled": True, "all_agents": False}}
    server_client.post = AsyncMock(side_effect=post)
    server = OpenCodeSandboxedAgent(config=_config(), server_client=server_client)
    downloads: list[str] = []
    sandbox = _sandbox(scenario, downloads)
    server._start_sandbox = AsyncMock(return_value=sandbox)

    result = await server.run(_RunRequest(), OpenCodeSandboxedAgentRunRequest.model_validate(RUN_BODY))

    run_command = sandbox.exec.await_args_list[0].kwargs["command"]
    config_arg = next(arg for arg in shlex.split(run_command) if arg.startswith("OPENCODE_CONFIG_CONTENT="))
    record = {
        "exec_calls": [call.kwargs for call in sandbox.exec.await_args_list],
        "opencode_config": json.loads(config_arg.removeprefix("OPENCODE_CONFIG_CONTENT=")),
        "downloads": downloads,
        "server_posts": posts,
        "result": result.model_dump(mode="json"),
    }
    # Random per run: the response id and creation time, the data-home suffix and the test's tmp dir.
    record["result"]["response"]["created_at"] = 0
    text = json.dumps(record).replace(str(tmp_path), "<tmp>")
    text = re.sub(r"nemo-gym-opencode-[0-9a-f]{32}", "nemo-gym-opencode-<hex>", text)
    text = re.sub(r"resp_[0-9a-f]{32}", "resp_<hex>", text)
    return json.loads(text)


@mark.parametrize("scenario", SCENARIOS)
async def test_run_matches_golden(scenario: str, tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    record = await _run(scenario, tmp_path, monkeypatch)

    golden = json.loads(GOLDEN_PATH.read_text()) if GOLDEN_PATH.exists() else {}
    if os.environ.get("NEMO_GYM_UPDATE_GOLDEN") == "1":
        golden[scenario] = record
        GOLDEN_PATH.write_text(json.dumps(golden, indent=2, sort_keys=True) + "\n")

    assert record == golden[scenario]


def test_run_endpoint_returns_the_opencode_response_model() -> None:
    server_client = MagicMock(spec=ServerClient)
    server_client.global_config_dict = {}
    server = OpenCodeSandboxedAgent(config=_config(), server_client=server_client)

    [run_route] = [route for route in server.setup_webserver().routes if getattr(route, "path", None) == "/run"]

    # FastAPI serializes /run through this model; a narrower one would drop the opencode_* fields.
    assert run_route.response_model is OpenCodeSandboxedAgentVerifyResponse
