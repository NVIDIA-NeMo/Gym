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

import asyncio
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

from nemo_gym.sandbox import SandboxExecResult, SandboxSpec, sandbox_spec_from_mapping
from responses_api_agents.claude_code_agent.sandbox_runner import ClaudeCodeSandboxRunner


BASE_REVISION = "a" * 40


def _result(stdout: str = "", stderr: str = "", return_code: int = 0, error_type: str | None = None):
    return SandboxExecResult(
        stdout=stdout,
        stderr=stderr,
        return_code=return_code,
        error_type=error_type,
    )


class FakeSandbox:
    def __init__(
        self,
        provider,
        spec,
        responses,
        *,
        download_contents: str = "",
        download_files: dict[str, str] | None = None,
        start_delay: float = 0,
        stop_error: Exception | None = None,
    ):
        self.provider = provider
        self.spec = spec
        self.responses = list(responses)
        self.started = False
        self.stopped = False
        self.uploaded_members: list[str] = []
        self.exec_calls: list[dict] = []
        self.download_contents = download_contents
        self.download_files = {
            "/tmp/nemo_gym_claude.stdout": "",
            "/tmp/nemo_gym_claude.stderr": "",
            "/tmp/nemo_gym_claude.status": "0",
            **(download_files or {}),
        }
        self.start_delay = start_delay
        self.stop_error = stop_error

    async def start(self):
        if self.start_delay:
            await asyncio.sleep(self.start_delay)
        self.started = True
        return self

    async def upload(self, local_path, remote_path):
        assert remote_path == "/tmp/nemo_gym_claude_config.tar.gz"
        with tarfile.open(local_path, mode="r:gz") as archive:
            self.uploaded_members = sorted(archive.getnames())

    async def exec(self, command, **kwargs):
        self.exec_calls.append({"command": command, **kwargs})
        return self.responses.pop(0)

    async def download(self, remote_path, local_path):
        if remote_path == "/tmp/nemo_gym_workspace.patch":
            contents = self.download_contents
        else:
            contents = self.download_files[remote_path]
        Path(local_path).write_text(contents)

    async def stop(self):
        self.stopped = True
        if self.stop_error is not None:
            raise self.stop_error


def _config_dir(tmp_path: Path) -> Path:
    config_dir = tmp_path / "config"
    (config_dir / "skills" / "example").mkdir(parents=True)
    (config_dir / "settings.json").write_text("{}")
    (config_dir / "skills" / "example" / "SKILL.md").write_text("---\nname: example\ndescription: Example.\n---\n")
    return config_dir


def _runner(fake: FakeSandbox, *, max_patch_bytes: int = 1024) -> ClaudeCodeSandboxRunner:
    return ClaudeCodeSandboxRunner(
        provider={"fake": {}},
        spec=SandboxSpec(image="test", workdir="/workspace/nemo-gym"),
        workspace="/workspace/nemo-gym",
        timeout_s=30,
        max_patch_bytes=max_patch_bytes,
        sandbox_factory=lambda provider, spec: fake,
    )


def test_run_stages_config_executes_agent_and_captures_patch(tmp_path: Path) -> None:
    patch = "diff --git a/probe.txt b/probe.txt\n"
    fake = FakeSandbox(
        {"fake": {}},
        SandboxSpec(),
        [
            _result(),  # unpack config
            _result(),  # clean workspace
            _result(f"{BASE_REVISION}\n"),
            _result(),
            _result(),  # git add -N
            _result(),  # write patch in sandbox
            _result("0"),  # git diff status
            _result(str(len(patch.encode()))),
        ],
        download_contents=patch,
        download_files={"/tmp/nemo_gym_claude.stdout": '{"type":"result"}\n'},
    )

    result = asyncio.run(
        _runner(fake).run(
            command=["claude", "-p", "--", "create probe.txt"],
            env={"ANTHROPIC_API_KEY": "secret"},  # pragma: allowlist secret
            config_dir=_config_dir(tmp_path),
        )
    )

    assert fake.started is True
    assert fake.stopped is True
    assert "settings.json" in fake.uploaded_members
    assert "skills/example/SKILL.md" in fake.uploaded_members
    assert result.stdout == '{"type":"result"}\n'
    assert result.workspace_patch == patch
    assert result.base_revision == BASE_REVISION
    agent_call = fake.exec_calls[3]
    assert agent_call["cwd"] == "/workspace/nemo-gym"
    assert agent_call["env"]["CLAUDE_CONFIG_DIR"] == "/tmp/nemo_gym_claude_config"
    assert "claude -p -- 'create probe.txt'" in agent_call["command"]
    assert "head -c 52428800" in agent_call["command"]
    assert "--no-textconv" in fake.exec_calls[5]["command"]
    assert "head -c 1025" in fake.exec_calls[5]["command"]
    assert BASE_REVISION in fake.exec_calls[5]["command"]


def test_cleanup_runs_when_setup_fails(tmp_path: Path) -> None:
    fake = FakeSandbox(
        {"fake": {}},
        SandboxSpec(),
        [_result(stderr="tar failed", return_code=2)],
    )

    with pytest.raises(RuntimeError, match="stage Claude Code configuration"):
        asyncio.run(
            _runner(fake).run(
                command=["claude", "-p", "--", "task"],
                env={},
                config_dir=_config_dir(tmp_path),
            )
        )

    assert fake.stopped is True


def test_cleanup_failure_does_not_mask_setup_failure(tmp_path: Path) -> None:
    fake = FakeSandbox(
        {"fake": {}},
        SandboxSpec(),
        [_result(stderr="tar failed", return_code=2)],
        stop_error=RuntimeError("stop failed"),
    )

    with pytest.raises(RuntimeError, match="stage Claude Code configuration") as exc_info:
        asyncio.run(
            _runner(fake).run(
                command=["claude", "-p", "--", "task"],
                env={},
                config_dir=_config_dir(tmp_path),
            )
        )

    assert any("cleanup also failed" in note for note in exc_info.value.__notes__)


def test_timeout_bounds_sandbox_start_and_still_cleans_up(tmp_path: Path) -> None:
    fake = FakeSandbox(
        {"fake": {}},
        SandboxSpec(),
        [],
        start_delay=0.05,
    )
    runner = ClaudeCodeSandboxRunner(
        provider={"fake": {}},
        spec=SandboxSpec(image="test", workdir="/workspace/nemo-gym"),
        workspace="/workspace/nemo-gym",
        timeout_s=0.01,
        sandbox_factory=lambda provider, spec: fake,
    )

    with pytest.raises(TimeoutError):
        asyncio.run(
            runner.run(
                command=["claude", "-p", "--", "task"],
                env={},
                config_dir=_config_dir(tmp_path),
            )
        )

    assert fake.stopped is True


def test_rejects_oversized_patch(tmp_path: Path) -> None:
    fake = FakeSandbox(
        {"fake": {}},
        SandboxSpec(),
        [
            _result(),
            _result(),
            _result(BASE_REVISION),
            _result(),
            _result(),
            _result(),
            _result("141"),
            _result("9"),
        ],
    )

    with pytest.raises(RuntimeError, match="exceeding the 4-byte limit"):
        asyncio.run(
            _runner(fake, max_patch_bytes=4).run(
                command=["claude", "-p", "--", "task"],
                env={},
                config_dir=_config_dir(tmp_path),
            )
        )

    assert fake.stopped is True


def test_rejects_git_diff_failure(tmp_path: Path) -> None:
    fake = FakeSandbox(
        {"fake": {}},
        SandboxSpec(),
        [
            _result(),
            _result(),
            _result(BASE_REVISION),
            _result(),
            _result(),
            _result(),
            _result("2"),
            _result("0"),
        ],
    )

    with pytest.raises(RuntimeError, match="git diff failed"):
        asyncio.run(
            _runner(fake).run(
                command=["claude", "-p", "--", "task"],
                env={},
                config_dir=_config_dir(tmp_path),
            )
        )


def test_does_not_capture_patch_after_command_timeout(tmp_path: Path) -> None:
    fake = FakeSandbox(
        {"fake": {}},
        SandboxSpec(),
        [
            _result(),
            _result(),
            _result(BASE_REVISION),
            _result(stderr="timed out", return_code=125, error_type="timeout"),
        ],
    )

    with pytest.raises(RuntimeError, match="did not complete safely"):
        asyncio.run(
            _runner(fake).run(
                command=["claude", "-p", "--", "task"],
                env={},
                config_dir=_config_dir(tmp_path),
            )
        )

    assert fake.stopped is True
    assert len(fake.exec_calls) == 4


def test_rejects_dirty_workspace_before_agent_runs(tmp_path: Path) -> None:
    fake = FakeSandbox(
        {"fake": {}},
        SandboxSpec(),
        [
            _result(),
            _result(" M existing.py\n?? leaked.txt\n"),
        ],
    )

    with pytest.raises(RuntimeError, match="must start clean"):
        asyncio.run(
            _runner(fake).run(
                command=["claude", "-p", "--", "task"],
                env={},
                config_dir=_config_dir(tmp_path),
            )
        )

    assert fake.stopped is True
    assert len(fake.exec_calls) == 2


def test_rejects_forbidden_skill_directories(tmp_path: Path) -> None:
    fake = FakeSandbox(
        {"fake": {}},
        SandboxSpec(),
        [
            _result(),
            _result(),
            _result(return_code=1),
        ],
    )
    runner = ClaudeCodeSandboxRunner(
        provider={"fake": {}},
        spec=SandboxSpec(image="test", workdir="/workspace/nemo-gym"),
        workspace="/workspace/nemo-gym",
        timeout_s=30,
        forbidden_workspace_paths=(".agents/skills", ".claude/skills"),
        sandbox_factory=lambda provider, spec: fake,
    )

    with pytest.raises(RuntimeError, match="forbidden auto-discovery paths"):
        asyncio.run(
            runner.run(
                command=["claude", "-p", "--", "task"],
                env={},
                config_dir=_config_dir(tmp_path),
            )
        )

    assert ".agents/skills" in fake.exec_calls[2]["command"]


def test_build_sandbox_spec_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="Unknown SandboxSpec keys: typo"):
        sandbox_spec_from_mapping({"image": "test", "typo": True}, default_workdir="/workspace")


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"entrypoint": "bash"}, "entrypoint must be a list of strings"),
        ({"ttl_s": 0}, "ttl_s must be greater than zero"),
        ({"env": ["NOT_A_MAPPING"]}, "env must be a mapping"),
    ],
)
def test_build_sandbox_spec_validates_field_types(config, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        sandbox_spec_from_mapping(config, default_workdir="/workspace")


def test_build_sandbox_spec_merges_metadata() -> None:
    spec = sandbox_spec_from_mapping(
        {
            "image": "test",
            "metadata": {"task": "fixture"},
            "resources": {"cpu": 2, "memory_mib": 1024},
        },
        default_workdir="/workspace",
        metadata={"agent": "claude"},
        default_metadata={"task": "provider-default", "provider": "docker"},
    )

    assert spec.image == "test"
    assert spec.workdir == "/workspace"
    assert spec.metadata == {"task": "fixture", "provider": "docker", "agent": "claude"}
    assert spec.resources.cpu == 2
    assert spec.resources.memory_mib == 1024


@pytest.mark.sandbox
def test_docker_provider_smoke_captures_applicable_patch(tmp_path: Path) -> None:
    image = os.getenv("NEMO_GYM_CLAUDE_CODE_SANDBOX_IMAGE")
    if not image:
        pytest.skip("set NEMO_GYM_CLAUDE_CODE_SANDBOX_IMAGE to run the Docker sandbox smoke test")

    runner = ClaudeCodeSandboxRunner(
        provider={"docker": {}},
        spec=SandboxSpec(image=image, workdir="/workspace/nemo-gym", ttl_s=120),
        workspace="/workspace/nemo-gym",
        timeout_s=60,
    )
    result = asyncio.run(
        runner.run(
            command=["sh", "-c", "printf sandbox-smoke > sandbox_probe.txt"],
            env={},
            config_dir=_config_dir(tmp_path),
        )
    )

    assert "sandbox_probe.txt" in result.workspace_patch
    apply_repo = tmp_path / "apply-repo"
    apply_repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=apply_repo, check=True)
    patch_path = tmp_path / "workspace.patch"
    patch_path.write_text(result.workspace_patch)
    subprocess.run(["git", "apply", "--check", str(patch_path)], cwd=apply_repo, check=True)
