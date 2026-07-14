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

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nemo_gym.openai_utils import (
    NeMoGymResponse,
    NeMoGymResponseCreateParamsNonStreaming,
    NeMoGymResponseOutputMessage,
    NeMoGymResponseOutputText,
)
from nemo_gym.sandbox import SandboxExecResult, SandboxSpec
from nemo_gym.server_utils import ServerClient
from resources_servers.agent_skills.app import (
    AgentSkillCheckSuiteConfig,
    AgentSkillsResourcesServer,
    AgentSkillsResourcesServerConfig,
    AgentSkillsVerifyRequest,
)
from resources_servers.agent_skills.verifier import PatchVerificationResult, SandboxPatchVerifier


BASE_REVISION = "a" * 40
PATCH = "diff --git a/probe.txt b/probe.txt\n"


def _result(stdout: str = "", stderr: str = "", return_code: int = 0, error_type: str | None = None):
    return SandboxExecResult(
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        error_type=error_type,
    )


class FakeSandbox:
    def __init__(self, responses, *, download_files: dict[str, str] | None = None):
        self.responses = list(responses)
        self.uploaded_patch = ""
        self.uploaded_files: dict[str, str] = {}
        self.exec_calls: list[str] = []
        self.exec_kwargs: list[dict] = []
        self.events: list[str] = []
        self.stopped = False
        self.download_files = download_files or {}

    async def start(self):
        return self

    async def upload(self, local_path, remote_path):
        contents = Path(local_path).read_text()
        self.uploaded_files[remote_path] = contents
        self.events.append(f"upload:{remote_path}")
        if remote_path == "/tmp/nemo_gym_submission.patch":
            self.uploaded_patch = contents

    async def exec(self, command, **kwargs):
        self.exec_calls.append(command)
        self.exec_kwargs.append(kwargs)
        self.events.append(f"exec:{command}")
        return self.responses.pop(0)

    async def download(self, remote_path, local_path):
        Path(local_path).write_text(self.download_files[remote_path])

    async def stop(self):
        self.stopped = True


def _response() -> NeMoGymResponse:
    return NeMoGymResponse(
        id="response",
        created_at=0,
        model="model",
        object="response",
        output=[
            NeMoGymResponseOutputMessage(
                id="message",
                role="assistant",
                type="message",
                content=[NeMoGymResponseOutputText(type="output_text", text="done", annotations=[])],
            )
        ],
        parallel_tool_calls=False,
        tool_choice="none",
        tools=[],
    )


def _request(**overrides) -> AgentSkillsVerifyRequest:
    data = {
        "responses_create_params": NeMoGymResponseCreateParamsNonStreaming(input="task"),
        "response": _response(),
        "verifier_metadata": {"task_id": "task-1", "check_suite_id": "suite"},
        "workspace_patch": PATCH,
        "workspace_base_revision": BASE_REVISION,
    }
    data.update(overrides)
    return AgentSkillsVerifyRequest.model_validate(data)


@pytest.mark.asyncio
async def test_patch_verifier_applies_patch_and_runs_hidden_check(tmp_path: Path) -> None:
    path_backed_check = tmp_path / "path_backed.py"
    path_backed_check.write_text("assert True\n")
    fake = FakeSandbox(
        [
            _result(),  # clean status
            _result(BASE_REVISION),
            _result(),  # git apply --check
            _result(),  # git apply
            _result(),  # create hidden check dir
            _result(),  # protect hidden check files
            _result(),  # bounded hidden check wrapper
        ],
        download_files={
            "/tmp/nemo_gym_hidden_check.stdout": "all checks passed",
            "/tmp/nemo_gym_hidden_check.stderr": "",
            "/tmp/nemo_gym_hidden_check.status": "0",
        },
    )
    verifier = SandboxPatchVerifier(
        provider={"fake": {}},
        spec=SandboxSpec(image="test"),
        workspace="/workspace/nemo-gym",
        check_command="pytest -q hidden_tests",
        timeout_s=30,
        hidden_files={"hidden_tests/test_probe.py": "def test_probe(): assert True\n"},
        hidden_file_paths={"hidden_tests/path_backed.py": path_backed_check},
        sandbox_factory=lambda provider, spec: fake,
    )

    result = await verifier.verify(patch=PATCH, expected_base_revision=BASE_REVISION)

    assert result.passed is True
    assert result.status == "pass"
    assert fake.uploaded_patch == PATCH
    assert "pytest -q hidden_tests" in fake.exec_calls[-1]
    assert fake.exec_kwargs[-1]["cwd"] == "/tmp/nemo_gym_hidden_checks"
    assert fake.exec_kwargs[-1]["user"] == "root"
    assert "su -s /bin/sh nobody" in fake.exec_calls[-1]
    assert "NEMO_GYM_WORKSPACE=/workspace/nemo-gym" in fake.exec_calls[-1]  # pragma: allowlist secret
    assert fake.uploaded_files["/tmp/nemo_gym_hidden_checks/hidden_tests/test_probe.py"].startswith("def test_probe")
    assert fake.uploaded_files["/tmp/nemo_gym_hidden_checks/hidden_tests/path_backed.py"] == "assert True\n"
    apply_event = next(
        index
        for index, event in enumerate(fake.events)
        if event == "exec:git apply --binary /tmp/nemo_gym_submission.patch"
    )
    hidden_upload_event = next(
        index
        for index, event in enumerate(fake.events)
        if event == "upload:/tmp/nemo_gym_hidden_checks/hidden_tests/test_probe.py"
    )
    assert hidden_upload_event > apply_event
    assert fake.stopped is True


@pytest.mark.asyncio
async def test_patch_verifier_rejects_fixture_revision_mismatch() -> None:
    fake = FakeSandbox([_result(), _result("b" * 40)])
    verifier = SandboxPatchVerifier(
        provider={"fake": {}},
        spec=SandboxSpec(image="test"),
        workspace="/workspace/nemo-gym",
        check_command="pytest",
        timeout_s=30,
        sandbox_factory=lambda provider, spec: fake,
    )

    with pytest.raises(ValueError, match="different base revisions"):
        await verifier.verify(patch=PATCH, expected_base_revision=BASE_REVISION)

    assert fake.stopped is True


@pytest.mark.parametrize(
    ("check_user", "expected"),
    [
        ("root", "NEMO_GYM_WORKSPACE=/workspace/nemo-gym pytest"),
        (65534, 'su -s /bin/sh "$(getent passwd 65534 | cut -d: -f1)"'),
    ],
)
def test_hidden_check_wrapper_supports_same_and_numeric_users(check_user, expected) -> None:
    verifier = SandboxPatchVerifier(
        provider={"fake": {}},
        spec=SandboxSpec(image="test"),
        workspace="/workspace/nemo-gym",
        check_command="pytest",
        timeout_s=30,
        user="root",
        check_user=check_user,
    )

    command = verifier._bounded_check_command(
        stdout_path="/tmp/stdout",
        stderr_path="/tmp/stderr",
        status_path="/tmp/status",
        stdout_pipe="/tmp/stdout.pipe",
        stderr_pipe="/tmp/stderr.pipe",
    )

    assert expected in command


def _server() -> AgentSkillsResourcesServer:
    config = AgentSkillsResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="agent_skills",
        sandbox_provider={"fake": {}},
        check_suites={
            "suite": AgentSkillCheckSuiteConfig(
                sandbox_spec={"image": "test"},
                workspace="/workspace/nemo-gym",
                check_command="pytest -q hidden_tests",
            )
        },
    )
    server = AgentSkillsResourcesServer(config=config, server_client=MagicMock(spec=ServerClient))
    server.server_client.global_config_dict = {}
    return server


@pytest.mark.asyncio
async def test_resources_server_returns_component_score() -> None:
    server = _server()
    verification = PatchVerificationResult(
        passed=True,
        status="pass",
        stdout=(
            '{"status":"summary","scores":{"task_success":1.0,"correctness":1.0,'
            '"completeness":0.75,"convention_compliance":1.0}}\n'
        ),
        stderr="",
        return_code=0,
        verifier_base_revision=BASE_REVISION,
        elapsed_seconds=2.5,
    )
    verifier = MagicMock()
    verifier.verify = AsyncMock(return_value=verification)

    with patch("resources_servers.agent_skills.app.SandboxPatchVerifier", return_value=verifier):
        result = await server.verify(_request())

    assert result.reward == 1.0
    assert result.correctness == 1.0
    assert result.completeness == 0.75
    assert result.convention_compliance == 1.0
    assert result.status == "pass"
    assert result.verifier_elapsed_seconds == 2.5
    assert '"status":"summary"' in result.details["stdout"]
    assert "workspace_patch" not in result.model_dump()


@pytest.mark.asyncio
async def test_resources_server_reward_uses_overall_task_success() -> None:
    server = _server()
    verification = PatchVerificationResult(
        passed=False,
        status="check_failed",
        stdout=(
            '{"status":"summary","scores":{"task_success":0.0,"correctness":1.0,'
            '"completeness":0.0,"convention_compliance":1.0}}\n'
        ),
        stderr="",
        return_code=1,
        verifier_base_revision=BASE_REVISION,
        elapsed_seconds=1.0,
    )
    verifier = MagicMock()
    verifier.verify = AsyncMock(return_value=verification)

    with patch("resources_servers.agent_skills.app.SandboxPatchVerifier", return_value=verifier):
        result = await server.verify(_request())

    assert result.reward == 0.0
    assert result.correctness == 1.0
    assert result.completeness == 0.0


@pytest.mark.asyncio
async def test_resources_server_rejects_unknown_suite_without_sandbox() -> None:
    server = _server()
    result = await server.verify(_request(verifier_metadata={"task_id": "task-1", "check_suite_id": "missing"}))

    assert result.reward == 0.0
    assert result.status == "unknown_check_suite"


@pytest.mark.asyncio
async def test_resources_server_converts_verifier_exception_to_failure() -> None:
    server = _server()
    verifier = MagicMock()
    verifier.verify = AsyncMock(side_effect=RuntimeError("sandbox unavailable"))

    with patch("resources_servers.agent_skills.app.SandboxPatchVerifier", return_value=verifier):
        result = await server.verify(_request())

    assert result.reward == 0.0
    assert result.status == "verifier_error"
    assert result.details["reason"] == "sandbox unavailable"


@pytest.mark.asyncio
async def test_resources_server_converts_missing_hidden_file_to_failure(tmp_path: Path) -> None:
    server = _server()
    server.config.check_suites["suite"].hidden_file_paths = {"run_checks.py": str(tmp_path / "missing.py")}

    result = await server.verify(_request())

    assert result.reward == 0.0
    assert result.status == "verifier_error"
    assert "do not exist" in result.details["reason"]
