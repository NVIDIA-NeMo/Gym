# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run a SWE resources server's seed_session and verify against a provider-level fake.

The real ``AsyncSandbox`` runs, so the network policy, ``sandbox_config.files``, ``setup_command`` and
the cleanup ``start_with_setup`` does on a failed setup are all exercised through the server's endpoints.
"""

import importlib
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi.testclient import TestClient
from pytest import MonkeyPatch

from nemo_gym.client.server_client import ServerClient
from nemo_gym.config import ModelServerRef
from nemo_gym.sandbox.api import AsyncSandbox
from nemo_gym.sandbox.providers.base import SandboxExecResult, SandboxHandle, SandboxSpec, SandboxStatus
from resources_servers.swebench.patch_capture import PatchCapture


SETUP_COMMAND = "chmod +x /tmp/wrapper.sh"
MODEL_URL = "http://10.1.2.3:8000"
AGENT_POLICY = {"defaultAction": "deny", "egress": [{"action": "allow", "target": "10.1.2.3"}]}
RESPONSE = {
    "output": [],
    "id": "",
    "created_at": 0,
    "model": "",
    "object": "response",
    "parallel_tool_calls": False,
    "tool_choice": "auto",
    "tools": [],
}


class _Provider:
    """Records every create, exec and close; ``setup_result`` answers the setup command."""

    name = "opensandbox"

    def __init__(self, setup_result: dict[str, SandboxExecResult]) -> None:
        self.setup_result = setup_result
        self.specs: dict[str, SandboxSpec] = {}
        self.commands: dict[str, list[str]] = {}
        self.closed: list[str] = []

    async def create(self, spec: SandboxSpec) -> SandboxHandle:
        sandbox_id = f"sb-{len(self.specs)}"
        self.specs[sandbox_id] = spec
        self.commands[sandbox_id] = []
        return SandboxHandle(sandbox_id=sandbox_id, provider_name=self.name, raw=None)

    async def exec(self, handle: SandboxHandle, command: str, **kwargs) -> SandboxExecResult:
        self.commands[handle.sandbox_id].append(command)
        if command == SETUP_COMMAND:
            kind = "agent" if "network_policy" in self.specs[handle.sandbox_id].provider_options else "evaluation"
            return self.setup_result.get(kind, SandboxExecResult(return_code=0, stdout="", stderr=""))
        # Seeding reads the base commit with `git rev-parse HEAD`, which must print something.
        return SandboxExecResult(return_code=0, stdout="seed-base\n", stderr="")

    async def upload_file(self, handle: SandboxHandle, source_path: Path, target_path: str) -> None:
        pass

    async def download_file(self, handle: SandboxHandle, source_path: str, target_path: Path) -> None:
        raise NotImplementedError

    async def status(self, handle: SandboxHandle) -> SandboxStatus:
        return SandboxStatus.RUNNING

    async def serialize_handle(self, handle: SandboxHandle, *, scope: str | None = None) -> dict:
        return {"provider": self.name, "sandbox_id": handle.sandbox_id}

    async def connect(self, descriptor) -> SandboxHandle:
        return SandboxHandle(sandbox_id=descriptor["sandbox_id"], provider_name=self.name, raw=None)

    async def close(self, handle: SandboxHandle) -> None:
        self.closed.append(handle.sandbox_id)

    async def aclose(self) -> None:
        pass


def check_swe_agent_sandbox(
    monkeypatch: MonkeyPatch,
    tmp_path: Path,
    module_name: str,
    prefix: str,
    *,
    entry: str,
    setup_result: dict[str, SandboxExecResult],
) -> None:
    """Seed through ``entry`` ("legacy" or "episode"), then verify; ``setup_result`` makes setup fail."""
    app = importlib.import_module(f"resources_servers.{module_name}.app")
    verification = importlib.import_module(f"resources_servers.{module_name}.verification")
    agent_tools = importlib.import_module("nemo_gym.agents.agent_tools")
    row = json.loads((Path(app.__file__).parent / "data/example.jsonl").read_text().splitlines()[0])
    row = row.get("verifier_metadata", row)
    provider = _Provider(setup_result)
    monkeypatch.setattr(app, "AsyncSandbox", lambda config: AsyncSandbox(provider, owns_provider=False))
    monkeypatch.setattr(app, "get_global_config_dict", lambda: {})
    monkeypatch.setattr(app, "resolve_provider_config", lambda *_: {"opensandbox": {}})
    monkeypatch.setattr(app, "resolve_provider_metadata", lambda *_: {})
    monkeypatch.setattr(agent_tools, "get_server_url", lambda name: MODEL_URL)
    monkeypatch.setattr(app, "prepare_git_for_commits", AsyncMock())
    monkeypatch.setattr(
        app, "capture_model_patch", AsyncMock(return_value=PatchCapture.static("patch", "worktree", "worktree"))
    )
    grade = AsyncMock(
        return_value=verification.VerificationResult(
            completed=True, resolved=True, patch_applied=True, test_results={}, test_output="log", error=None
        )
    )
    monkeypatch.setattr(app, "run_verification", grade)
    if hasattr(app, "resolve_parser"):
        monkeypatch.setattr(app, "resolve_parser", lambda *args: MagicMock())
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text("#!/bin/sh\n")
    server = getattr(app, prefix + "ResourcesServer")(
        config=getattr(app, prefix + "ResourcesServerConfig")(
            host="0.0.0.0",
            port=8080,
            entrypoint="",
            name=module_name,
            sandbox_provider="opensandbox",
            sandbox_config={"files": {"/tmp/wrapper.sh": str(wrapper)}, "setup_command": SETUP_COMMAND},
            apply_anti_cheating=False,
            verification_log_dir=None,
            enforce_agent_no_network=True,
            sandbox_model_server=ModelServerRef(type="responses_api_models", name="policy_model"),
        ),
        server_client=MagicMock(spec=ServerClient),
    )
    seed = {
        "resources_session_id": "session",
        "episode_id": {"rollout_id": "rollout"},
        "task_id": {"taskset": module_name, "task_id": row["instance_id"]},
        "task_data": row,
    }
    verify = row | {"responses_create_params": {"input": "original prompt"}, "response": RESPONSE}
    # A seed expected to succeed raises the server's own error, so a failure shows its traceback.
    with TestClient(server.setup_webserver(), raise_server_exceptions="agent" not in setup_result) as client:
        seeded = client.post("/seed_session", json=row if entry == "legacy" else seed)
        agent = "sb-0"
        assert provider.specs[agent].provider_options["network_policy"] == AGENT_POLICY
        assert provider.specs[agent].files["/tmp/wrapper.sh"] == "#!/bin/sh\n"
        assert provider.commands[agent][0] == SETUP_COMMAND
        if "agent" in setup_result:
            # A failed setup stops the agent sandbox and the seed keeps nothing.
            assert seeded.status_code != 200
            assert provider.closed == [agent]
            assert not server._session_id_to_sandbox
            return
        assert seeded.status_code == 200, seeded.text
        result = client.post("/verify", json=verify)
        assert result.status_code == 200, result.text
        body = result.json()
        evaluation = [sandbox_id for sandbox_id in provider.specs if sandbox_id != agent]
        for sandbox_id in evaluation:
            assert "network_policy" not in provider.specs[sandbox_id].provider_options
            assert provider.specs[sandbox_id].files["/tmp/wrapper.sh"] == "#!/bin/sh\n"
            assert provider.commands[sandbox_id][0] == SETUP_COMMAND
        if "evaluation" in setup_result:
            # Every attempt's setup fails: retried on a fresh sandbox, each stopped, then incomplete and masked.
            assert len(evaluation) == 1 + server.config.inconclusive_verification_retries
            assert set(evaluation) <= set(provider.closed)
            assert body["evaluation_completed"] is False and body["mask_sample"] is True
            assert body["failure_kind"]
            grade.assert_not_awaited()
        else:
            assert len(evaluation) == 1
            assert body["evaluation_completed"] is True
            grade.assert_awaited_once()
        if entry == "episode":
            close = {"resources_session_id": "session", "episode_id": {"rollout_id": "rollout"}}
            assert client.post("/close_session", json=close).status_code == 200
    if entry == "episode":
        assert set(provider.specs) == set(provider.closed)
