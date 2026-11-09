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

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from nemo_gym.client import SESSION_ID_KEY, ServerClient
from nemo_gym.core import EpisodeId, TaskId
from nemo_gym.sandbox import SandboxExecResult
from nemo_gym.server_runtime import ResourcesSeedSessionRequest
from resources_servers.swebench_pro.app import SWEBenchProResourcesServerConfig
from resources_servers.swebench_pro_v2.app import (
    SWEBenchProV2InstanceRequest,
    SWEBenchProV2ResourcesServer,
    SWEBenchProV2SeedSessionRequest,
)
from resources_servers.swebench_pro_v2.task_data import TaskData
from resources_servers.swebench_pro_v2.verification import TEST_FILES, VerificationInputsV2, assemble_workspace_files


MODEL_POLICY = {"model_server": {"type": "responses_api_models", "name": "policy_model"}}
DIGEST = "sha256:" + "a" * 64


def instance_request() -> SWEBenchProV2InstanceRequest:
    return SWEBenchProV2InstanceRequest.model_validate(
        {
            "repo": "example/repo",
            "instance_id": "instance_example",
            "base_commit": "HEAD",
            "patch": "gold patch",
            "test_patch": "test patch",
            "problem_statement": "Fix it",
            "fail_to_pass": '["new_test"]',
            "pass_to_pass": '["old_test"]',
            "selected_test_files_to_run": '["tests"]',
            "dockerhub_tag": "example-tag",
            "docker_image": "ghcr.io/scaleapi/swe-bench_pro-v2:instance_example",
            "image_digest": DIGEST,
            "run_script": "#!/bin/bash\nexit 0\n",
            "parser_script": "print('{}')\n",
            "test_sh": "#!/bin/bash\nexit 0\n",
            "test_config": '{"fail_to_pass": "[]"}',
        }
    )


def make_server(**overrides: object) -> SWEBenchProV2ResourcesServer:
    config = SWEBenchProResourcesServerConfig.model_validate(
        {
            "host": "127.0.0.1",
            "port": 8080,
            "entrypoint": "",
            "name": "swebench_pro_v2_resources_server",
            "sandbox_provider": "test",
            "sandbox_config": {"provider_options": {"resource_requests": {"cpu": 1}}},
            "apply_anti_cheating": False,
            "environment_repairs": [],
        }
        | overrides
    )
    return SWEBenchProV2ResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))


def test_workspace_preserves_the_pinned_task_files_and_patch() -> None:
    inputs = make_server()._verification_inputs(instance_request(), "agent patch\r\n")
    files, _ = assemble_workspace_files(inputs.instance_id, None, inputs.patch, inputs.__dict__)

    assert set(TEST_FILES) <= set(files)
    assert files["test.sh"] == inputs.test_sh
    assert files["config.json"] == inputs.test_config
    assert files["test_patch.patch"] == inputs.test_patch
    assert files["patch.diff"] == "agent patch\r\n"
    assert isinstance(inputs, VerificationInputsV2)


@pytest.mark.asyncio
@pytest.mark.parametrize("task", [False, True])
async def test_only_the_agent_sandbox_is_network_locked_and_both_use_the_pinned_image(
    monkeypatch: pytest.MonkeyPatch, task: bool
) -> None:
    config = {"test": {"opensandbox": {}}, "head_server": {"host": "127.0.0.1"}}
    monkeypatch.setattr("resources_servers.swebench_pro.app.get_global_config_dict", lambda: config)
    monkeypatch.setattr("resources_servers.swebench_pro_v2.app.get_global_config_dict", lambda: config)
    monkeypatch.setattr("nemo_gym.agents.agent_tools.get_server_url", lambda name: "http://model.example:8000")
    sandbox = SimpleNamespace(start=AsyncMock())
    monkeypatch.setattr("resources_servers.swebench_pro.app.AsyncSandbox", lambda provider: sandbox)
    server = make_server(task_network_policy=MODEL_POLICY)

    await server._create_sandbox(instance_request(), task=task)

    spec = sandbox.start.await_args.args[0]
    assert spec.image == f"ghcr.io/scaleapi/swe-bench_pro-v2@{DIGEST}"
    expected = {"resource_requests": {"cpu": 1}}
    if task:
        expected["network_policy"] = {
            "defaultAction": "deny",
            "egress": [{"action": "allow", "target": "model.example"}],
        }
    assert spec.provider_options == expected
    assert server.config.sandbox_config["provider_options"] == {"resource_requests": {"cpu": 1}}


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["0.0.0.0", "::", "10.0.0.1", "head.example"])
async def test_network_lock_refuses_a_head_that_the_agent_could_reach(
    monkeypatch: pytest.MonkeyPatch, host: str
) -> None:
    monkeypatch.setattr(
        "resources_servers.swebench_pro_v2.app.get_global_config_dict", lambda: {"head_server": {"host": host}}
    )
    create = MagicMock()
    monkeypatch.setattr("resources_servers.swebench_pro.app.AsyncSandbox", create)

    with pytest.raises(ValueError, match="head server on loopback"):
        await make_server(task_network_policy=MODEL_POLICY)._create_sandbox(instance_request(), task=True)

    create.assert_not_called()


@pytest.mark.parametrize("field", ["docker_image", "image_digest", "test_sh", "test_config"])
@pytest.mark.parametrize("bad", [None, 1])
def test_schema_and_runtime_reject_missing_or_non_string_task_assets(field: str, bad: object) -> None:
    row = instance_request().model_dump() | {"evaluator_commit": "upstream", "dataset_revision": "dataset"}
    for model in (SWEBenchProV2InstanceRequest, TaskData):
        with pytest.raises(ValueError, match=field):
            model.model_validate(row | {field: bad})
        with pytest.raises(ValueError, match=field):
            model.model_validate({key: value for key, value in row.items() if key != field})
        assert model.model_fields[field].is_required()


class LocalGitSandbox:
    """Execute the resource server's real git commands in a temporary task repository."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.stopped = False
        self._handle = SimpleNamespace(sandbox_id="local-task")

    def path(self, remote: str) -> Path:
        return self.root / remote.removeprefix("/")

    async def exec(self, command: str, **kwargs: object) -> SandboxExecResult:
        local = command
        for remote in ("/testbed", "/app", "/tmp"):
            local = local.replace(remote, str(self.path(remote)))
        completed = subprocess.run(local, shell=True, executable="/bin/bash", capture_output=True, text=True)
        stdout = completed.stdout.replace(str(self.root), "")
        return SandboxExecResult(stdout=stdout, stderr=completed.stderr, return_code=completed.returncode)

    async def download(self, remote: str, local: Path) -> None:
        shutil.copyfile(self.path(remote), local)

    async def stop(self) -> None:
        self.stopped = True

    async def serialize(self) -> dict[str, str]:
        return {"sandbox_id": "local-task"}


@pytest.mark.asyncio
@pytest.mark.parametrize("repository", ["app", "testbed"])
@pytest.mark.parametrize("typed", [False, True])
async def test_agent_handoff_snapshot_and_patch_capture_use_the_task_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, repository: str, typed: bool
) -> None:
    for directory in ("app", "testbed", "tmp"):
        (tmp_path / directory).mkdir()
    repo = tmp_path / repository
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.com"], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "Test"], check=True)
    (repo / "source.txt").write_text("before\n")
    subprocess.run(["git", "-C", str(repo), "add", "source.txt"], check=True)
    subprocess.run(["git", "-C", str(repo), "-c", "commit.gpgsign=false", "commit", "-qm", "initial"], check=True)
    (repo / "image-artifact.log").write_text("already in the image\n")
    sandbox = LocalGitSandbox(tmp_path)
    server = make_server()
    monkeypatch.setattr(server, "_create_sandbox", AsyncMock(return_value=sandbox))
    row = instance_request().model_dump()
    if typed:
        seed = ResourcesSeedSessionRequest(
            resources_session_id="session",
            episode_id=EpisodeId(rollout_id="rollout"),
            task_id=TaskId(taskset="swebench_pro_v2", task_id="instance_example"),
            task_data=row,
        )
    else:
        seed = SWEBenchProV2SeedSessionRequest.model_validate(row)

    response = await server.seed_session(SimpleNamespace(session={SESSION_ID_KEY: "session"}), seed)
    assert (response.sandbox_access.workdir if typed else response.workdir) == f"/{repository}"
    (repo / "source.txt").write_text("after\n")
    (repo / "new-source.txt").write_text("new\n")

    patch = await server._extract_model_patch("session", "HEAD")

    assert "+after\n" in patch
    assert "+new\n" in patch
    assert "image-artifact.log" not in patch
    assert sandbox.stopped
    assert not server._session_id_to_sandbox


def test_verifier_attempt_can_cover_allocation_full_tests_and_artifact_collection() -> None:
    import yaml

    from resources_servers.swebench_pro_v2.verification import EXEC_MARGIN_S

    root = Path(__file__).parents[3]
    document = yaml.safe_load((root / "resources_servers/swebench_pro_v2/configs/swebench_pro_v2.yaml").read_text())
    config = SWEBenchProResourcesServerConfig.model_validate(
        document["swebench_pro_v2_resources_server"]["resources_servers"]["swebench_pro_v2"]
        | {"host": "127.0.0.1", "port": 8080, "name": "swebench_pro_v2_resources_server"}
    )

    assert config.verification_attempt_timeout > (
        config.sandbox_config["ready_timeout_s"] + config.evaluation_timeout + EXEC_MARGIN_S
    )
    assert config.verification_total_timeout >= (
        config.verification_attempt_timeout * (1 + config.inconclusive_verification_retries)
    )
