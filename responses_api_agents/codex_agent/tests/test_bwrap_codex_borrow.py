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

"""codex_agent borrows a bwrap sandbox a resources server created.

The real ``AsyncSandbox.connect``, descriptor and host-side file transfer run; only the
bwrap process itself is emulated (no namespaces on the test host): installer/prepare
commands succeed, and the supervised Codex launch publishes its artifacts into the
sandbox's private /tmp the way the in-sandbox supervisor and runner would.
"""

import asyncio
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from omegaconf import OmegaConf

from nemo_gym.agents import AgentSeedSessionRequest
from nemo_gym.client import ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxSpec, SandboxStatus
from nemo_gym.sandbox.providers.bwrap import BwrapProvider
from responses_api_agents.codex_agent.app import CodexAgent, CodexAgentConfig


pytestmark = pytest.mark.sandbox


def _events() -> str:
    items = [
        {"type": "item.completed", "item": {"id": "msg-1", "type": "agent_message", "text": "Submitted"}},
        {"type": "turn.completed", "usage": {"input_tokens": 11, "output_tokens": 4, "cached_input_tokens": 0}},
    ]
    return "\n".join(json.dumps([float(i), event]) for i, event in enumerate(items))


@pytest.fixture
def providers(tmp_path: Path):
    bwrap = tmp_path / "bwrap"
    bwrap.write_text("#!/bin/sh\nexit 99\n")  # never run: the receiver's process layer is emulated
    bwrap.chmod(0o755)
    usr = tmp_path / "usr"
    usr.mkdir()
    config = {
        "bwrap_path": str(bwrap),
        "base_dir": str(tmp_path / "base"),
        "ro_binds": [str(usr)],
        "network": "host",
        "network_host_ro_binds": [],
        "probe_on_create": False,
    }
    return BwrapProvider(**config), BwrapProvider(**config)


async def test_codex_agent_borrows_a_bwrap_sandbox(providers, tmp_path: Path) -> None:
    creator, receiver = providers
    # The resources server's side: create, seed task files, serialize.
    owned = await AsyncSandbox(creator, owns_provider=False).start(
        SandboxSpec(workdir="/workspace", files={"/workspace/submit.sh": "#!/bin/sh\n", "/workspace/AGENTS.md": "x"})
    )
    descriptor = json.loads(json.dumps(await owned.serialize()))
    inst = owned._handle.raw

    calls: list[tuple[list[str], dict[str, str]]] = []

    async def emulated_run(sandbox, argv, env, timeout):
        calls.append((argv, env))
        command = argv[-1]
        if "exec bash " in command and "process_supervisor.sh" in command:
            (params,) = (sandbox.tmp / "nemo-gym-codex-sessions").glob("*/params.sh")
            directory = f"/tmp/nemo-gym-codex-sessions/{params.parent.name}"
            receipt = {"cleanup_confirmed": True, "return_code": 0, "error": None, "timed_out": False}
            for name, text in (
                ("cleanup.json", json.dumps(receipt)),
                ("runtime.json", json.dumps({"hostname": "node", "pid": 2})),
                ("events.jsonl", _events()),
            ):
                await asyncio.to_thread(receiver._write_bytes, sandbox, f"{directory}/{name}", text.encode(), 0o644)
        return SandboxExecResult(stdout="", stderr="", return_code=0)

    receiver._run = emulated_run
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = OmegaConf.create(
        {"policy": {"responses_api_models": {"openai_model": {"host": "model.example", "port": 9000}}}}
    )
    client._build_server_base_url.return_value = "http://model.example:9000"
    agent_config = CodexAgentConfig(
        name="codex",
        host="localhost",
        port=8001,
        entrypoint="app.py",
        num_workers=1,
        model_server={"type": "responses_api_models", "name": "policy"},
        model="test-model",
        codex_version="0.144.4",
        session_close_timeout_seconds=3,
    )
    seed = AgentSeedSessionRequest(
        agent_session_id="codex-bwrap-session",
        episode_id=EpisodeId(rollout_id="codex-bwrap", attempt=1),
        task_id=TaskId(taskset="example", task_id="task"),
        sandbox_access={
            "connection": {"kind": "direct", "provider_config_ref": "sandbox", "descriptor": descriptor},
            "workdir": "/workspace",
        },
    )
    module = "responses_api_agents.codex_agent.app"

    def run_agent() -> dict:
        with (
            patch(f"{module}.ensure_codex", side_effect=AssertionError("native sessions must not install host Codex")),
            patch(f"{module}.resolve_provider_config"),
            patch(f"{module}.get_global_config_dict", return_value={}),
            patch(f"{module}.create_provider", return_value=receiver),
        ):
            agent = CodexAgent(config=agent_config, server_client=client)
            with TestClient(agent.setup_webserver()) as http:
                created = http.post("/v1/agent_sessions", json=seed.model_dump(mode="json"))
                assert created.status_code == 200, created.text
                session_id = created.json()["agent_session_id"]
                result = http.post("/ng-rollout/codex-bwrap-a1/v1/responses", json={"input": "Write a kernel"})
                assert result.status_code == 200, result.text
                closed = http.post(
                    "/v1/agent_sessions/close",
                    json={"agent_session_id": session_id, "episode_id": seed.episode_id.model_dump()},
                )
                assert closed.status_code == 200, closed.text
                return result.json()

    response = await asyncio.to_thread(run_agent)
    assert response["status"] == "completed"
    assert response["usage"]["total_tokens"] == 15

    # Codex's files went through host-side transfer into the sandbox's private /tmp and home.
    (session,) = (inst.tmp / "nemo-gym-codex-sessions").iterdir()
    for name in (
        "install_codex_runtime.sh",
        "harness_runner.sh",
        "process_supervisor.sh",
        "params.sh",
        "home/.codex/config.toml",
    ):
        assert (session / name).is_file(), name
    assert (session / "prompt.txt").read_text() == "Write a kernel"
    assert "cwd=/workspace" in (session / "params.sh").read_text()

    # Every exec ran over the borrowed sandbox's directories, GPUs hidden; the agent's own work runs in the
    # workdir, and control steps (stop, session-dir removal) may run from / (codex_agent's control_cwd).
    assert len(calls) >= 4  # prepare, install, supervised launch, session-dir removal
    for argv, env in calls:
        assert "--unshare-net" not in argv  # host network: Codex reaches the model server
        assert argv[argv.index(str(inst.workspace)) + 1] == "/workspace"
        assert argv[argv.index("--chdir") + 1] in ("/workspace", "/")
        assert env["NVIDIA_VISIBLE_DEVICES"] == "void" and env["HOME"] == "/home/sandbox"
    launches = [argv for argv, _ in calls if "process_supervisor.sh" in argv[-1] and "exec bash " in argv[-1]]
    assert len(launches) == 1
    assert launches[0][launches[0].index("--chdir") + 1] == "/workspace"

    # Borrowing never stops the sandbox; its owner does.
    assert await owned.status() == SandboxStatus.RUNNING
    assert (inst.workspace / "submit.sh").read_text() == "#!/bin/sh\n"
    await owned.stop()
    assert not inst.root.exists()
